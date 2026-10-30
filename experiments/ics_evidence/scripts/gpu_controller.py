"""Inference queue: idle GPUs, with explicitly permitted memory-limited sharing on GPU0/1."""
import json
import os
import subprocess
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from evidence_cpu import ROOT, DOMAINS, dump, sha

PY = '/data1/liuyeqiang/damot_rebuttal_20261026/env/bin/python'
SCRIPT = ROOT/'code/tools/analysis/ics_probe_extract.py'
VK = ['clone','fog','morning','overcast','rain','sunset']
CROOT = Path('/data1/liuyeqiang/ics_v2_20261030_184026')
FCK = '/data1/liuyeqiang/damot_rebuttal_20261026/cache/weights/F_stage4_e5.pth'
HCK = '/data1/liuyeqiang/damot_apr_complement_hf_20261028_1420/episode/adapt/stage04_after_night/epoch_5.pth'
GPU_LOCK=threading.Lock()
RESERVED=set()
for manifest in (ROOT/'manifests').glob('infer_*.json'):
    record=json.loads(manifest.read_text())
    if record.get('status')=='running' and Path('/proc/%d'%record['pid']).exists():
        RESERVED.add(record['gpu'])


def acquire(preferred,name):
    while True:
        with GPU_LOCK:
            lines=subprocess.check_output(['nvidia-smi',
                '--query-gpu=index,memory.free,memory.used,utilization.gpu,memory.total',
                '--format=csv,noheader,nounits']).decode().strip().splitlines()
            stats={int(a[0]):tuple(map(int,a[1:])) for a in ([v.strip() for v in line.split(',')] for line in lines)}
            # Newly authorized idle cards take priority over shared occupied cards.
            idle=[g for g,(free,used,util,total) in stats.items() if used<256 and util<=10]
            order=sorted(idle,key=lambda g:(g!=preferred,g))+[g for g in [1,0] if g not in idle]
            for gpu in order:
                if gpu in RESERVED:continue
                free,used,util,total=stats[gpu]
                if gpu in idle or (gpu==1 and free>=16000) or (gpu==0 and free>=5000):
                    RESERVED.add(gpu)
                    fraction=min(.35,(free-1536)/total)
                    return gpu,fraction
        print('WAIT_GPU',name,'idle cards or memory-limited sharing0/1 unavailable',flush=True)
        time.sleep(20)


def release(gpu):
    with GPU_LOCK:RESERVED.discard(gpu)


def params(label, checkpoint, views, bdd=False, smoke=False):
    args=[PY,'-u',str(SCRIPT),'--checkpoint',str(checkpoint),'--label',label,
          '--out-dir',str(ROOT/('smoke_feats' if smoke else 'features')),'--views',views,
          '--dose-frames-per-video','0','--classes','car','--box-source','gt',
          '--max-videos','1' if smoke else '0','--frames-per-video','1' if smoke else ('30' if bdd else '60'),
          '--frame-stride','1','--max-rois-per-frame','100000',
          '--min-box','0','--img-scale','1296' if bdd else '1242','720' if bdd else '375',
          '--view-batch-size','4' if bdd else '1','--workers','2' if bdd else '1','--seed','0','--memory-fraction','.35',
          '--allow-uninitialized-bank']
    if bdd:
        for domain in DOMAINS:
            prefix=Path('/data1/liuyeqiang/ics_v2_20261030_184026/data_views')/domain
            if not prefix.exists():
                prefix=Path('/data1/liuyeqiang/damot_rebuttal_20261026/data/views')/domain
            args+=['--domain',domain,str(ROOT/'data/native_BDD_fit'/(domain+'.json')),str(prefix)]
    else:
        for domain in (['clone'] if smoke else VK):
            args+=['--domain',domain,str(ROOT/'data/vkitti2/annotations'/(domain+'.json')),str(ROOT/'data/vkitti2/images')]
        args+=['--video-regex',r'(?P<scene>Scene\d+)_(?P<variant>\w+)_Camera_\d+']
    return args


def job(label, checkpoint, views, gpu, bdd=False, smoke=False):
    name=label+('_smoke' if smoke else '')
    complete=ROOT/'manifests'/('infer_'+name+'.json')
    if complete.exists() and json.loads(complete.read_text()).get('status')=='complete':
        return
    if complete.exists() and json.loads(complete.read_text()).get('status')=='running':
        record=json.loads(complete.read_text());pid=record['pid']
        print('REATTACH',name,pid,flush=True)
        while Path('/proc/%d'%pid).exists():
            try:stat=Path('/proc/%d/stat'%pid).read_text().split()
            except FileNotFoundError:break
            if stat[2]=='Z':break
            time.sleep(10)
        final=ROOT/('smoke_feats' if smoke else 'features')/label/'meta.json'
        assert final.exists(), 'Detached inference exited without a complete feature manifest: '+name
        record.update(status='complete',end=time.time(),returncode=0,resumed_monitor=True)
        dump(complete,record)
        release(record['gpu'])
        return
    args=params(label,checkpoint,views,bdd,smoke)
    gpu,fraction=acquire(gpu,name)
    args[args.index('--memory-fraction')+1]=str(fraction)
    if fraction<.25:
        args[args.index('--view-batch-size')+1]='1'
        args[args.index('--workers')+1]='1'
    env=dict(os.environ,CUDA_VISIBLE_DEVICES=str(gpu),OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',
             MKL_NUM_THREADS='1',PYTHONDONTWRITEBYTECODE='1',TMPDIR=str(ROOT/'tmp'))
    start=time.time()
    before=subprocess.check_output(['nvidia-smi','--query-compute-apps=gpu_uuid,pid,used_memory','--format=csv,noheader']).decode()
    print('START',name,'GPU',gpu,flush=True)
    with (ROOT/'logs'/('infer_'+name+'.log')).open('w') as log:
        proc=subprocess.Popen(args,cwd=str(ROOT/'code'),env=env,stdout=log,stderr=subprocess.STDOUT)
        dump(complete,dict(status='running',gpu=gpu,pid=proc.pid,command=args,start=start,
             checkpoint=str(checkpoint),checkpoint_sha256=sha(checkpoint),pre_existing_gpu_apps=before))
        rc=proc.wait()
    manifest=json.loads(complete.read_text())
    manifest.update(status='complete' if rc==0 else 'failed',returncode=rc,end=time.time())
    dump(complete,manifest)
    release(gpu)
    if rc:raise RuntimeError((name,rc))
    print('DONE',name,'seconds',time.time()-start,flush=True)


def model_part2(label,ck,gpu):
    job(label,ck,'clean,fap,photo',gpu,smoke=True)
    job(label,ck,'clean,fap,photo',gpu)


def main():
    while not (ROOT/'manifests/vkitti2_stage_complete.json').exists():time.sleep(10)
    assert json.loads((ROOT/'manifests/vkitti2_stage_complete.json').read_text())['status']=='complete'
    with ThreadPoolExecutor(max_workers=2) as pool:
        futures=[pool.submit(model_part2,'F',FCK,0),pool.submit(model_part2,'HF',HCK,1)]
        for f in futures:f.result()
    native=ROOT/'weights/VK_self_stage5_e5.pth'
    if native.exists():
        assert native.stat().st_size==682303345, 'native checkpoint copy incomplete'
        with ThreadPoolExecutor(max_workers=2) as pool:
            fs=[pool.submit(job,'VK_self_BDD',native,'clean,fap,photo,student',0,True),
                pool.submit(model_part2,'VK_self',native,1)]
            for f in fs:f.result()
    dump(ROOT/'manifests/part2_inference_complete.json',dict(status='complete',time=time.time()))
    # Third part is dispatched only after every formal C run has ended its training.
    while not all((CROOT/'runs'/g/'v2_end_rank0.json').exists() for g in ['C0','C1','C2','C3','C4']):
        print('WAIT_C_TRAINING',flush=True)
        time.sleep(30)
    tasks=[]
    for group in ['C0','C1','C2','C3','C4']:
        tasks.append((group,CROOT/'runs'/group/'episode/adapt/stage04_after_night/epoch_2.pth'))
    tasks.append(('X3-1',Path('/data1/liuyeqiang/ics_quick_20261029_160807/X3/X3-1/episode/adapt/stage04_after_night/epoch_2.pth')))
    with ThreadPoolExecutor(max_workers=6) as pool:
        fs=[pool.submit(job,group,ck,'clean',[4,7,8,9,1,0][i]) for i,(group,ck) in enumerate(tasks)]
        for f in fs:f.result()
    dump(ROOT/'manifests/part3_inference_complete.json',dict(status='complete',time=time.time()))


if __name__=='__main__':
    main()
