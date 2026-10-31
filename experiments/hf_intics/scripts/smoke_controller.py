import csv
import json
import os
from pathlib import Path
import socket
import subprocess
import sys
import time
Q=Path(__file__).resolve().parents[1]
PY='/data1/liuyeqiang/damot_rebuttal_20261026/env/bin/python'

def dump(name,obj):
    path=Q/'manifests'/name
    tmp=path.with_suffix(path.suffix+'.tmp')
    tmp.write_text(json.dumps(obj,indent=2)+'\n');tmp.replace(path)

def completed(variant):
    root=Q/'preflight/smoke'/variant
    try:
        for rank in range(4):
            summary=json.loads((root/('summary_rank%d.json'%rank)).read_text())
            rows=list(csv.DictReader((root/('iterations_rank%d.csv'%rank)).open()))
            assert summary['status']=='complete' and summary['iterations']==50
            assert summary['seed']==1790355234 and len(rows)==50
            assert [int(r['iteration']) for r in rows]==list(range(1,51))
            assert all(r['loss_track_trim'] is not None for r in rows)
        return True
    except (FileNotFoundError,ValueError,AssertionError,KeyError):
        return False

def job(name,args,count):
    dump('smoke_controller.json',dict(status='running',phase=name,pid=os.getpid(),time=time.time()))
    command=[sys.executable,str(Q/'scripts/gpu_job.py'),'--name',name,'--gpus',str(count),
        '--prefer','2,3,5,6,4,7,8,9','--']+args
    with (Q/'logs'/(name+'.log')).open('w') as f:
        result=subprocess.run(command,stdout=f,stderr=subprocess.STDOUT)
    if result.returncode:
        dump('smoke_controller.json',dict(status='failed',phase=name,exit_code=result.returncode))
        raise SystemExit(result.returncode)

assert json.loads((Q/'preflight/U1.json').read_text())['status']=='passed'
assert (Q/'preflight/U2.json').exists()
assert json.loads((Q/'preflight/input_storage_checks.json').read_text())['status']=='passed'
assert json.loads((Q/'preflight/NAS_storage_checks.json').read_text())['status']=='passed'
# Recheck the disabled branch against the same saved original after the final
# geometry-only assertion was added to the enabled branch.
off_job=Q/'manifests/U1_final_off_job.json'
if not off_job.exists() or json.loads(off_job.read_text())['status']!='complete':
    job('U1_final_off',[PY,str(Q/'scripts/preflight_batch.py'),'--mode','off'],1)
for variant in ['HF','INTICS']:
    if completed(variant):
        print('REUSE_COMPLETE_SMOKE',variant,flush=True)
        continue
    with socket.socket() as sock:
        sock.bind(('',0));port=sock.getsockname()[1]
    job('smoke_'+variant,[PY,'-m','torch.distributed.launch','--nproc_per_node=4',
        '--master_port='+str(port),'--use_env',str(Q/'scripts/smoke.py'),
        '--variant',variant],4)

records={}
for variant in ['HF','INTICS']:
    root=Q/'preflight/smoke'/variant
    summaries=[json.loads((root/('summary_rank%d.json'%r)).read_text()) for r in range(4)]
    rows=[list(csv.DictReader((root/('iterations_rank%d.csv'%r)).open())) for r in range(4)]
    assert all(s['iterations']==50 and len(rr)==50 for s,rr in zip(summaries,rows))
    critical=[max(float(rows[r][i]['iteration_seconds']) for r in range(4)) for i in range(10,50)]
    compute=[max(float(rows[r][i]['compute_seconds']) for r in range(4)) for i in range(10,50)]
    records[variant]=dict(summaries=summaries,key_ratio=sum(s['key_effective_batches'] for s in summaries)/200,
        ref_ratio=sum(s['ref_effective_batches'] for s in summaries)/200,
        iteration_mean_last40=sum(critical)/40,compute_mean_last40=sum(compute)/40,
        loss_track_trim_mean=sum(s['loss_track_trim_mean'] for s in summaries)/4)
new=records['INTICS'];old=records['HF']
passed=all(s['key_effective_batches']>=40 and s['ref_effective_batches']>=40 and
           s['first_initialized_iteration'] is not None for s in new['summaries'])
report=dict(status='passed' if passed else 'failed',records=records,seed=1790355234,
    relative_iteration_time_increase_pct=100*(new['iteration_mean_last40']/old['iteration_mean_last40']-1),
    relative_compute_time_increase_pct=100*(new['compute_mean_last40']/old['compute_mean_last40']-1),
    timing_protocol='four V100s, source S, same seed, batches/workers16/2, actual HF runner; max-rank time then mean iterations11-50; HF logs local, INTICS logs on NAS after disk-full restart; physical GPU IDs recorded per job',
    storage_migration='manifests/NAS_migration.json; completed HF50 reused, incomplete new-mode attempt archived and restarted from source S with the same seed',
    all_weights_disposable=True)
(Q/'preflight/U3.json').write_text(json.dumps(report,indent=2)+'\n')
dump('preflight.json',dict(status='passed' if passed else 'failed',U1='passed',
    U2='reported; no angle gate',U3=report['status'],seed=1790355234))
dump('smoke_controller.json',dict(status='complete' if passed else 'failed_gate',time=time.time()))
print('U3_COMPLETE',report['status'],flush=True)
