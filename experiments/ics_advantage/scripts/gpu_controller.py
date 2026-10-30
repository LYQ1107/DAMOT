"""Finite inference queue; acquire only completely idle GPUs, never share."""
import json,os,subprocess,threading,time
from concurrent.futures import ThreadPoolExecutor
from common import *

LOCK=threading.Lock();RESERVED=set()
EXTRACT=CODE/'tools/analysis/ics_probe_extract.py'

def alive(pid):
    try:
        return Path('/proc/%d/stat'%pid).read_text().split()[2]!='Z' and 'ics_probe_extract.py' in Path('/proc/%d/cmdline'%pid).read_text()
    except FileNotFoundError:return False

for existing in (ROOT/'manifests').glob('infer_*.json'):
    d=json.loads(existing.read_text())
    if d.get('status')=='running' and alive(d['pid']):RESERVED.add(d['gpu'])

def acquire(name):
    while True:
        with LOCK:
            data=subprocess.check_output(['nvidia-smi','--query-gpu=index,memory.used,utilization.gpu,memory.free,memory.total','--format=csv,noheader,nounits']).decode()
            policy_file=ROOT/'manifests/resource_policy.json'
            policy=json.loads(policy_file.read_text()) if policy_file.exists() else {}
            stats={}
            for line in data.strip().splitlines():
                g,used,util,free,total=map(int,line.split(','));stats[g]=(used,util,free,total)
            available=[g for g,(used,util,free,total) in stats.items() if used<256 and util<=10 and g not in RESERVED]
            if policy.get('allow_shared_0_1',False):
                # Sharing needs the explicit human resource-policy reply. Keep all
                # four planned training cards available under that policy.
                pending_training=not ((NEW/'episode/train_complete.json').exists() or (NEW/'episode/train_stopped.json').exists())
                shared=[g for g in [0,1] if g not in RESERVED and stats[g][2]>=12000]
                if pending_training:
                    available=shared+[g for g in available if g not in [0,1,2,3,5,6]]
                else:
                    available=sorted(available)+[g for g in shared if g not in available]
            if available:
                gpu=available[0] if policy.get('allow_shared_0_1',False) else sorted(available,key=lambda g:(g not in [2,3],g))[0]
                RESERVED.add(gpu)
                return gpu,data
        print('WAIT_IDLE_GPU',name,flush=True);time.sleep(20)

def args(label,ck,extra=False,vk=False,smoke=False):
    online=label.startswith(('HF_stage','INT_stage'))
    script=ROOT/'code_online/tools/analysis/ics_probe_extract.py' if online else EXTRACT
    command=[PY,'-u',str(script),'--checkpoint',str(ck),'--label',label,
       '--out-dir',str(ROOT/('smoke_features' if smoke else 'vk_features' if vk else 'tsne_features' if extra else 'features')),
       '--views','clean' if extra or vk or smoke else 'clean,fap,photo',
       '--box-source','gt','--allow-uninitialized-bank','--memory-fraction','.35',
       '--max-videos','1' if smoke else '0','--frames-per-video','1' if smoke else '60' if vk else '100000' if extra else '30','--frame-stride','1',
       '--dose-frames-per-video','0' if vk or smoke else '100000' if extra else '5','--max-rois-per-frame','100000','--min-box','0',
       '--img-scale','1242' if vk else '1296','375' if vk else '720','--view-batch-size','3','--workers','2','--seed','0']
    if vk:
        previous=Path('/data1/liuyeqiang/ics_evidence_20261030_215311/data/vkitti2')
        for d in ['clone','fog','morning','overcast','rain','sunset']:
            command+=['--domain',d,str(previous/'annotations'/(d+'.json')),str(previous/'images')]
        command+=['--video-regex',r'(?P<scene>Scene\d+)_(?P<variant>\w+)_Camera_\d+']
    elif extra:
        command+=['--dose-views','dark_0.40,fog_0.45,fapk_0.50','--domain','night',str(ROOT/'inputs/tsne_night.json'),str(image_prefix('night'))]
    else:
        for d in DOMAINS:command+=['--domain',d,str(ROOT/'inputs'/('probe_'+d+'.json')),str(image_prefix(d))]
    return command

def job(model,extra=False,vk=False,smoke=False):
    name=model+('_smoke' if smoke else '_vk' if vk else '_tsne' if extra else '')
    manifest=ROOT/'manifests'/('infer_'+name+'.json')
    if manifest.exists() and json.loads(manifest.read_text()).get('status')=='complete':return
    if manifest.exists() and json.loads(manifest.read_text()).get('status')=='running':
        record=json.loads(manifest.read_text());print('REATTACH',name,record['pid'],flush=True)
        while alive(record['pid']):time.sleep(10)
        final=ROOT/('smoke_features' if smoke else 'vk_features' if vk else 'tsne_features' if extra else 'features')/model/'meta.json'
        assert final.exists() and final.stat().st_mtime>=record['start'],('Detached extraction did not complete',name)
        assert json.loads(final.read_text())['checkpoint']==record['checkpoint']
        record.update(status='complete',returncode=0,end=time.time(),resumed_monitor=True,completion_basis='final manifest written after all requested domains')
        dump(manifest,record)
        with LOCK:RESERVED.discard(record['gpu'])
        return
    ck=checkpoint(model);command=args(model,ck,extra,vk,smoke)
    digest=sha(ck)
    gpu,stats=acquire(name);e=env();e['CUDA_VISIBLE_DEVICES']=str(gpu)
    print('START',name,'GPU',gpu,flush=True)
    with (ROOT/'logs'/('infer_'+name+'.log')).open('w') as f:
        working=ROOT/'code_online' if model.startswith(('HF_stage','INT_stage')) else CODE
        p=subprocess.Popen(command,cwd=str(working),env=e,stdout=f,stderr=subprocess.STDOUT)
        record=dict(status='running',pid=p.pid,gpu=gpu,command=command,checkpoint=str(ck),checkpoint_sha256=digest,start=time.time(),pre_start_gpu_status=stats)
        dump(manifest,record);rc=p.wait()
    record.update(status='complete' if rc==0 else 'failed',returncode=rc,end=time.time());dump(manifest,record)
    with LOCK:RESERVED.discard(gpu)
    if rc:raise RuntimeError((name,rc))
    print('DONE',name,flush=True)

def model_job(model):
    job(model);job(model,True)

def main():
    for extra in [False,True]:
        with (ROOT/'logs'/('dryrun_tsne.log' if extra else 'dryrun.log')).open('w') as f:
            subprocess.run(args('C0',checkpoint('C0'),extra)+['--dry-run'],env=env(True),stdout=f,stderr=subprocess.STDOUT,check=True)
    with ThreadPoolExecutor(max_workers=2) as pool:
        fs=[pool.submit(model_job,m) for m in ['C0','C2','C4','C1','C3']]
        for f in fs:f.result()
    dump(ROOT/'manifests/batch1_inference_complete.json',dict(status='complete',models=MODELS,time=time.time()))

if __name__=='__main__':main()
