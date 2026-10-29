"""Durable S1-S5 queue. Scientific failures stop their dependent work."""
import fcntl,hashlib,io,json,os,socket,subprocess,sys,tarfile,time
from pathlib import Path
from protocol import write_json
ROOT=Path(__file__).resolve().parents[1];PY=str(ROOT/'env/bin/python');CODE=ROOT/'code'
HF=Path('/data1/liuyeqiang/damot_apr_complement_hf_20261028_1420');NG=Path('/data1/liuyeqiang/damot_apr_complement_hf_nogeom_20261029_1443')
D=['overcast','rainy','dawn_dusk','night'];children={};failures={}
def terminal_success(name):
    if name.startswith('CPU_'):return (ROOT/'manifests'/('analysis_'+name[4:]+'_complete.json')).exists()
    if name.startswith('S5_probeCPU_'):return (ROOT/'results'/('analysis_S5_HF_'+name[-1]+'_v2')/'summary.json').exists()
    return False
class RecoveredProcess:
    def __init__(self,name,pid):self.name=name;self.pid=pid
    def poll(self):
        marker=ROOT/'manifests'/(self.name+'_gpu.json')
        if marker.exists():
            m=json.loads(marker.read_text())
            if m.get('status') in ['complete','failed']:return m['exit_code']
        if terminal_success(self.name):return 0
        proc=Path('/proc/%d/cmdline'%self.pid)
        try:raw=proc.read_bytes()
        except OSError:return 1
        return None if str(ROOT).encode() in raw else 1
def recover_running():
    for meta in (ROOT/'manifests').glob('*_process.json'):
        m=json.loads(meta.read_text());name=meta.name[:-len('_process.json')]
        if 'exit_code' in m:
            if m['exit_code']:failures[name]=dict(exit_code=m['exit_code'],log=m['log'])
            continue
        p=RecoveredProcess(name,m['pid'])
        if p.poll() is None:
            children[name]=(p,Path(m['log']).open('a'),Path(m['log']),m['command']);print('ADOPT',name,m['pid'],flush=True)
def process_matches(pattern):
    for p in Path('/proc').iterdir():
        if not p.name.isdigit() or int(p.name)==os.getpid():continue
        try:raw=(p/'cmdline').read_bytes().replace(b'\0',b' ').decode()
        except (OSError,UnicodeError):continue
        if pattern in raw and ('python' in raw) and 'bash -' not in raw:return True
    return False
def start(name,cmd,active=None,gpu=0,prefer='8,9,7,2,3,4,5,6',after=None):
    if (ROOT/'manifests'/('hold_'+name+'.json')).exists():return
    if name.startswith('S4_') and any(n.startswith('S4_') for n in children):return
    if name in failures or name in children or (active and process_matches(active)):return
    if gpu:
        prefix=[sys.executable,str(ROOT/'scripts/gpu_slot.py'),'--name',name,'--count',str(gpu),'--prefer',prefer]
        if after:prefix+=['--after',str(after)]
        cmd=prefix+['--']+cmd
    stamp=str(int(time.time()*1e9));log=ROOT/'logs'/(name+'_'+stamp+'.log');stream=log.open('w')
    payload={str(f.relative_to(ROOT)):f.read_bytes() for parent in [ROOT/'code',ROOT/'scripts'] for f in sorted(parent.rglob('*.py')) if '__pycache__' not in str(f)}
    hashes={k:hashlib.sha256(v).hexdigest() for k,v in payload.items()};digest=hashlib.sha256(json.dumps(hashes,sort_keys=True).encode()).hexdigest()
    archive=ROOT/'manifests/source_snapshots'/(digest+'.tar.gz');archive.parent.mkdir(exist_ok=True)
    if not archive.exists():
        with tarfile.open(str(archive),'w:gz') as tar:
            for rel,raw in sorted(payload.items()):
                info=tarfile.TarInfo(rel);info.size=len(raw);info.mode=0o644;info.mtime=0;tar.addfile(info,io.BytesIO(raw))
    write_json(ROOT/'manifests'/(name+'_code_'+stamp+'.json'),dict(files=hashes,source_archive=str(archive),source_archive_sha256=hashlib.sha256(archive.read_bytes()).hexdigest(),source_HF_snapshot=str(HF/'code'),publication_base_commit='a66f900ea093f9333ea66e4e70693e190d2ed2d8',time=time.time()))
    env=dict(os.environ,OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2',MKL_THREADING_LAYER='GNU',PYTHONUNBUFFERED='1')
    p=subprocess.Popen(cmd,cwd=str(CODE),env=env,stdout=stream,stderr=subprocess.STDOUT,start_new_session=True)
    children[name]=(p,stream,log,cmd);write_json(ROOT/'manifests'/(name+'_process.json'),dict(pid=p.pid,command=cmd,log=str(log),start=time.time()));print('START',name,p.pid,flush=True)
def features(name):return all((ROOT/'results/feats'/('%s_stage%d_e5'%(name,k))/'meta.json').exists() for k in range(1,5))
def s4_done(name):return all((ROOT/'results/S4'/name/tag/d/'complete.json').exists() for tag in ['baseline']+['%s_beta%g'%(b,v) for b in ['ICS_stored_m2','FAP_clear_fit_m2','Identity_fit_m16','Random_m2'] for v in [.5,1.]] for d in D)
def port():
    s=socket.socket();s.bind(('127.0.0.1',0));n=s.getsockname()[1];s.close();return n
def main():
    lock=(ROOT/'.controller.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    recover_running()
    write_json(ROOT/'manifests/controller.json',dict(pid=os.getpid(),start=time.time(),NOGEOM_release_marker=str(NG/'controller_complete.json')))
    while True:
        for name,(p,stream,log,cmd) in list(children.items()):
            code=p.poll()
            if code is None:continue
            stream.close();del children[name]
            write_json(ROOT/'manifests'/(name+'_process.json'),dict(pid=p.pid,exit_code=code,command=cmd,log=str(log),end=time.time()))
            if code:failures[name]=dict(exit_code=code,log=str(log));print('FAILED',name,code,log,flush=True)
            else:print('DONE',name,flush=True)
        for name,source in [('HF',HF),('NOGEOM',NG)]:
            ready=name=='HF' or (NG/'controller_complete.json').exists()
            if ready and not features(name):start('S1_'+name,[PY,str(ROOT/'scripts/run_probe_group.py'),name,str(source),'{GPU}'],active='scripts/run_probe_group.py '+name,gpu=1)
            if features(name) and not (ROOT/'manifests'/('analysis_'+name+'_complete.json')).exists():start('CPU_'+name,[PY,str(ROOT/'scripts/analyze_group.py'),name])
        if not s4_done('F'):start('S4_F',[PY,'-u',str(CODE/'tools/run/ics_tracking_intervention.py'),'--model','F'],active='ics_tracking_intervention.py --model F',gpu=1)
        if s4_done('F') and not (ROOT/'results/S4/F/native_projection_validation.json').exists():start('S4_native_F',[PY,str(CODE/'tools/run/ics_validate_native_projection.py'),'--model','F'],gpu=1)
        if (ROOT/'manifests/hold_S4_F.json').exists():
            captured=all((ROOT/'cache/S4/HF'/d/'complete.json').exists() for d in D)
            if not captured:start('S4_HF_capture',[PY,'-u',str(CODE/'tools/run/ics_tracking_intervention.py'),'--model','HF','--capture-only'],gpu=1)
            elif not all((ROOT/'results/S4/HF/baseline'/d/'complete.json').exists() for d in D):start('S4_HF_baseline',[PY,'-u',str(CODE/'tools/run/ics_tracking_intervention.py'),'--model','HF','--baseline-only'],gpu=1)
        HF_baseline_done=all((ROOT/'results/S4/HF/baseline'/d/'complete.json').exists() for d in D)
        if (ROOT/'results/S4/F/native_projection_validation.json').exists() and not HF_baseline_done:start('S4_HF_baseline',[PY,'-u',str(CODE/'tools/run/ics_tracking_intervention.py'),'--model','HF','--baseline-only'],gpu=1)
        if (ROOT/'results/S4/F/native_projection_validation.json').exists() and HF_baseline_done and (ROOT/'results/analysis_HF_stage4_e5_v2/bases/provenance.json').exists() and not s4_done('HF'):start('S4_HF',[PY,'-u',str(CODE/'tools/run/ics_tracking_intervention.py'),'--model','HF'],active='ics_tracking_intervention.py --model HF',gpu=1)
        if s4_done('HF') and not (ROOT/'results/S4/HF/native_projection_validation.json').exists():start('S4_native_HF',[PY,str(CODE/'tools/run/ics_validate_native_projection.py'),'--model','HF'],gpu=1)
        # NOGEOM evaluation finishes before these four GPUs are eligible.
        for ix,g in enumerate('abcd'):
            out=ROOT/'results/S5'/('HF_'+g)
            previous=ix==0 or (ROOT/'results/S5'/('HF_'+'abcd'[ix-1])/'train_complete.json').exists()
            if (ROOT/'results/S4/HF/native_projection_validation.json').exists() and previous and not (out/'train_complete.json').exists():
                cmd=[PY,'-m','torch.distributed.launch','--nproc_per_node=4','--master_port='+str(port()),'--use_env',str(CODE/'tools/run/ics_short_stage.py'),'--base-config',str(out/'config.py'),'--source-ckpt','/data1/liuyeqiang/damot_rebuttal_20261026/cache/weights/S.pth','--target-order']+D+['--eval-domains']+D+['--work-dir',str(out/'episode'),'--epochs-per-stage','2','--samples-per-gpu','16','--workers-per-gpu','2','--launcher','pytorch','--seed','1790355234','--resume-episode-from-stage','4','--no-validate']
                start('S5_train_'+g,cmd,gpu=4,prefer='3,4,5,6',after=NG/'controller_complete.json')
            if not (out/'train_complete.json').exists():continue
            # One evaluation worker per group, serial full-domain cells.
            if not all((out/'results/stage4'/d/'complete.json').exists() for d in D):start('S5_eval_'+g,[PY,str(ROOT/'scripts/s5_eval_group.py'),g],gpu=1)
            if not (ROOT/'results/feats'/('S5_HF_'+g)/'meta.json').exists():start('S5_probe_'+g,[PY,str(ROOT/'scripts/s5_probe.py'),g,'{GPU}'],gpu=1)
            elif not (ROOT/'results'/('analysis_S5_HF_'+g+'_v2')/'summary.json').exists():start('S5_probeCPU_'+g,[PY,str(ROOT/'scripts/s5_probe_analyze.py'),g])
        observed={label:process_matches(pattern) for label,pattern in [('adopted_S1_HF','scripts/run_probe_group.py HF'),('adopted_S4_F','ics_tracking_intervention.py --model F')]}
        write_json(ROOT/'manifests/controller_status.json',dict(running=list(children),observed=observed,failures=failures,time=time.time(),features={n:features(n) for n in ['HF','NOGEOM']},S4={n:s4_done(n) for n in ['F','HF']},NOGEOM_complete=(NG/'controller_complete.json').exists()))
        finished=all(features(n) and (ROOT/'manifests'/('analysis_'+n+'_complete.json')).exists() for n in ['HF','NOGEOM']) and all(s4_done(n) and (ROOT/'results/S4'/n/'native_projection_validation.json').exists() for n in ['F','HF']) and all((ROOT/'results'/('analysis_S5_HF_'+g+'_v2')/'summary.json').exists() and all((ROOT/'results/S5'/('HF_'+g)/'results/stage4'/d/'complete.json').exists() for d in D) for g in 'abcd')
        if finished and not children:write_json(ROOT/'manifests/experiments_complete.json',dict(status='complete',time=time.time()));print('ALL EXPERIMENTS COMPLETE',flush=True);return
        time.sleep(30)
if __name__=='__main__':main()
