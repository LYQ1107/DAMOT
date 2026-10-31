"""Wait for the independent training task, then run only frozen evaluation work."""
import json,shutil,subprocess,time,traceback
from concurrent.futures import ThreadPoolExecutor
from common import *

def state(status,**values):
    path=ROOT/'manifests/batch2_status.json';record=dict(status=status,source_root=str(NEW),**values)
    if path.exists() and read_json(path)==record:return
    dump(path,record);print('BATCH2_STATE',status,flush=True)

def source_readiness():
    formal=NEW/'manifests/formal_controller.json'
    try:status=read_json(formal).get('status')
    except (FileNotFoundError,json.JSONDecodeError):
        return 'waiting_source_manifest_replacement',dict(source_status='temporarily_unreadable_NAS_manifest')
    one=(NEW/'results/stage1/overcast/complete.json').exists()
    final=all((NEW/'results/stage4'/d/'complete.json').exists() for d in ['overcast','rainy','dawn_dusk','night'])
    if (NEW/'episode/train_complete.json').exists() and one and final and status in ['complete_awaiting_publication','complete','published']:
        return 'all_stages_ready',dict(source_status=status)
    if (NEW/'episode/train_stopped.json').exists() and one:
        gate=read_json(NEW/'episode/stage1_gate.json')
        return 'stage1_only_source_stopped',dict(source_status=status,gate=gate)
    if status in ['failed','preflight_failed_no_formal_training']:
        return 'waiting_source_failure_resolution',dict(source_status=status)
    return 'waiting_source_training_and_evaluation',dict(source_status=status)

def prepare_models(mode):
    snapshot=ROOT/'code_online'
    if not snapshot.exists():shutil.copytree(str(NEW/'code'),str(snapshot),ignore=shutil.ignore_patterns('__pycache__','.git'))
    extractor=snapshot/'tools/analysis/ics_probe_extract.py';extractor.parent.mkdir(parents=True,exist_ok=True)
    shutil.copy2(str(CODE/'tools/analysis/ics_probe_extract.py'),str(extractor))
    stages=[4,1] if mode=='all_stages_ready' else [1];records={}
    for stage in stages:
        domains=['overcast','rainy','dawn_dusk','night'] if stage==4 else ['overcast']
        for prefix,run in [('HF',HF),('INT',NEW)]:
            eval_roots={d:run/'results'/('stage%d'%stage)/d for d in domains}
            first=readcsv(next(iter(eval_roots.values()))/'metrics.csv')[0]
            ck=Path(first['checkpoint']);assert ck.is_file(),('Recorded final checkpoint unavailable',ck)
            digest=sha(ck)
            assert digest==first['checkpoint_sha256']
            for source in eval_roots.values():
                assert (source/'complete.json').exists()
                assert all(r['checkpoint_sha256']==digest for r in readcsv(source/'metrics.csv'))
            label=prefix+'_stage%d'%stage
            records[label]=dict(checkpoint=str(ck),checkpoint_sha256=digest,stage=stage,epoch=5,seed=1790355234,
                 role='online intervention-residual ICS' if prefix=='INT' else 'original HF ICS',
                 frozen_evaluations={d:str(p) for d,p in eval_roots.items()},inference_source_snapshot=str(snapshot))
    dump(ROOT/'manifests/batch2_model_provenance.json',records)
    dump(ROOT/'manifests/batch2_source_snapshot.json',dict(mode=mode,source=str(NEW/'code'),
         files=[dict(path=str(p.relative_to(snapshot)),sha256=sha(p)) for p in sorted(snapshot.rglob('*.py')) if '__pycache__' not in str(p)],
         extractor_same_as_batch1=sha(extractor)==sha(CODE/'tools/analysis/ics_probe_extract.py')))
    candidates=[NEW/'config_diff.json',NEW/'manifests/config_diff.json',NEW/'configs/config_diff.json']
    diffs=[]
    for p in candidates:
        if p.exists():diffs.append(dict(path=str(p),sha256=sha(p),content=json.loads(p.read_text())))
    dump(ROOT/'manifests/batch2_training_config_differences.json',diffs)
    return records

def feature_model(model):
    from gpu_controller import job
    job(model,smoke=True);job(model);job(model,extra=True);job(model,vk=True)

def tracking_model(model,domain,source):
    with (ROOT/'logs'/('tracking_'+model+'_'+domain+'.log')).open('w') as f:
        subprocess.run([PY,'-u',str(ROOT/'scripts/tracking_per_video.py'),'--model',model,'--domain',domain,'--source',source],env=env(True),stdout=f,stderr=subprocess.STDOUT,check=True)

def run_script(script,args=()):
    with (ROOT/'logs'/('batch2_'+script[:-3]+'.log')).open('w') as f:
        subprocess.run([PY,'-u',str(ROOT/'scripts'/script)]+list(args),env=env(True),stdout=f,stderr=subprocess.STDOUT,check=True)

def main():
    completed=ROOT/'manifests/batch2_analysis_complete.json'
    previous=read_json(completed) if completed.exists() else None
    if previous and previous['stage4_available']:return
    while True:
        try:mode,info=source_readiness()
        except OSError as exc:
            state('waiting_source_storage',error=str(exc));time.sleep(30);continue
        if mode=='stage1_only_source_stopped' and previous:
            state('waiting_stage4_after_stage1_supplement',stage1_supplement_complete=True,**info)
            time.sleep(30);continue
        state(mode,**info)
        if mode in ['all_stages_ready','stage1_only_source_stopped'] and (ROOT/'manifests/publication_batch1.json').exists():break
        time.sleep(30)
    records=prepare_models(mode);state('frozen_inference_and_native_sequence_metrics',mode=mode,models=list(records))
    # Four source training cards have been released at normal completion. Source
    # stopped runs likewise perform no training here; the resource policy still applies.
    with ThreadPoolExecutor(max_workers=4) as gpu_pool,ThreadPoolExecutor(max_workers=2) as cpu_pool:
        gf=[gpu_pool.submit(feature_model,m) for m in records]
        cf=[cpu_pool.submit(tracking_model,m,d,source) for m,r in records.items() for d,source in r['frozen_evaluations'].items()]
        for f in cf:f.result()
        run_script('tracking_statistics.py',['--batch','2'])
        for f in gf:f.result()
    dump(ROOT/'manifests/batch2_inference_complete.json',dict(status='complete',mode=mode,models=list(records)))
    state('offline_feature_statistics',mode=mode,models=list(records))
    run_script('feature_statistics.py',['--batch','2']);run_script('feature_figures.py',['--batch','2']);run_script('vkitti_statistics.py')
    dump(ROOT/'manifests/batch2_analysis_complete.json',dict(status='complete' if mode=='all_stages_ready' else 'available_stage1_complete_source_stopped',
        mode=mode,models=list(records),stage4_available=mode=='all_stages_ready',source_stop=info.get('gate')))
    run_script('report.py')
    state('ready_for_review_and_publication',mode=mode,models=list(records))
    if mode=='stage1_only_source_stopped':return main()

if __name__=='__main__':
    try:main()
    except BaseException:
        state('failed',error=traceback.format_exc());raise
