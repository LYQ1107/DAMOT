"""Wait for external training; read frozen stage4 only, then infer and evaluate."""
import fcntl,subprocess,time,traceback
from concurrent.futures import ThreadPoolExecutor,as_completed
from common import *
DOMAINS4=['overcast','rainy','dawn_dusk','night']
def state(status,**extra):
    path=ROOT/'manifests/batch2_status.json';value=dict(status=status,source_root=str(NEW),**extra)
    if path.exists() and read_json(path)==value:return False
    dump(path,value);print('BATCH2_STATE',status,flush=True)
    return True

def run(script,args,gpu=None,logname=None):
    path=ROOT/'logs'/(logname or ('batch2_'+script+'.log'))
    with path.open('ab',buffering=0) as log:
        subprocess.run([PY,'-u',str(ROOT/'scripts'/script)]+list(args),env=environment(gpu),stdin=subprocess.DEVNULL,stdout=log,stderr=log,check=True)

def acquire_gpu():
    while True:
        raw=subprocess.check_output(['nvidia-smi','--query-gpu=index,memory.used,memory.free,utilization.gpu','--format=csv,noheader,nounits'],universal_newlines=True)
        info={int(v[0]):tuple(map(int,v[1:])) for v in [r.split(',') for r in raw.strip().splitlines()]}
        for gpu in [2,3,5,6,7,9,0,1,4,8]:
            used,free,util=info[gpu]
            if (gpu not in [0,1] and (used>200 or util>5)) or free<12000:continue
            lock=Path('/tmp/ics_trim_gpu%d.lock'%gpu).open('w')
            try:fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
            except BlockingIOError:lock.close();continue
            return gpu,lock
        time.sleep(15)

def inference_job(label,domain,record):
    if (ROOT/'jobs'/label/domain/'inference_complete.json').exists():return
    gpu,lock=acquire_gpu()
    try:
        args=['--model',label,'--domain',domain,'--checkpoint',record['checkpoint'],
              '--source',record['evaluations'][domain],'--cases-json',record['cases_file']]
        run('infer.py',args+['--preflight'],gpu,'batch2_%s_%s_preflight.log'%(label,domain))
        run('infer.py',args,gpu,'batch2_%s_%s_infer.log'%(label,domain))
    finally:lock.close()

def metric_job(label,domain,source):
    run('metrics.py',['--label',label,'--domain',domain,'--source',source],logname='batch2_%s_%s_metrics.log'%(label,domain))

def main():
    if (ROOT/'analysis/batch2/complete.json').exists():return
    while True:
        formal=NEW/'manifests/formal_controller.json'
        try:status=read_json(formal).get('status')
        except (FileNotFoundError,json.JSONDecodeError):status='not_dispatched'
        ready=(NEW/'episode/train_complete.json').exists() and all((NEW/'results/stage4'/d/'complete.json').exists() for d in DOMAINS4)
        if ready and status in ['complete_awaiting_publication','complete','published','complete_published']:break
        if (NEW/'episode/train_stopped.json').exists():
            changed=state('waiting_stage4_after_source_gate',source_status=status,stop=read_json(NEW/'episode/train_stopped.json'),
                          scope='Original stage4 only; no stage1 inference-trim substitute',user_instruction='保留原范围：注明 stage4 缺失，等训练任务处理后再跑第二批')
            if changed:run('analysis.py',['--report'])
            time.sleep(45);continue
        if state('waiting_source_training_and_frozen_stage4',source_status=status):run('analysis.py',['--report'])
        time.sleep(45)
    records={}
    for label,base,beta in [('INT',NEW,.25),('HF',HF,.1)]:
        ev={d:base/'results/stage4'/d for d in DOMAINS4}
        first=rows(ev[DOMAINS4[0]]/'metrics.csv')[0];ck=Path(first['checkpoint']);checksum=sha(ck)
        assert checksum==first['checkpoint_sha256']
        for path in ev.values():
            assert (path/'complete.json').exists()
            assert all(r['checkpoint_sha256']==checksum and int(r['stage'])==4 and int(r['epoch'])==5 for r in rows(path/'metrics.csv'))
        casespec=[dict(label=label+'_b0',beta=0.,basis_source='checkpoint_bank',npy_path=None),
                  dict(label=label+('_b025' if label=='INT' else '_b01'),beta=beta,basis_source='checkpoint_bank',npy_path=None)]
        specfile=ROOT/'inputs'/('batch2_'+label+'_cases.json');dump(specfile,casespec)
        records[label]=dict(checkpoint=str(ck),checkpoint_sha256=checksum,cases_file=str(specfile),
                            evaluations={d:str(p) for d,p in ev.items()},seed=1790355234,stage=4,epoch=5)
        run('bank_audit.py',['--checkpoint',str(ck),'--label',label],logname='bank_audit_'+label+'.log')
    dump(ROOT/'manifests/batch2_model_provenance.json',records);state('running_frozen_inference',models=list(records))
    # Completed full training releases its cards; other cards must be wholly idle.
    with ThreadPoolExecutor(max_workers=4) as pool:
        futures=[pool.submit(inference_job,label,d,record) for label,record in records.items() for d in DOMAINS4]
        for future in as_completed(futures):future.result()
    state('running_offline_evaluation')
    with ThreadPoolExecutor(max_workers=2) as pool:
        futures=[pool.submit(metric_job,case['label'],d,record['evaluations'][d]) for label,record in records.items()
                 for case in json.loads(Path(record['cases_file']).read_text()) for d in DOMAINS4]
        for future in as_completed(futures):future.result()
    run('analysis.py',['--batch','2']);state('ready_for_review_and_publication')

if __name__=='__main__':
    try:main()
    except Exception as exc:
        state('failed',error=str(exc),traceback=traceback.format_exc());raise
