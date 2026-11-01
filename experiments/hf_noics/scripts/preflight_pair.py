"""Four cards leased for the same-seed HF/NoICS 50-update validation pair."""
import csv
import hashlib
import json
import math
import os
from pathlib import Path
import socket
import subprocess
import sys
import time
import traceback
Q=Path(__file__).resolve().parents[1]
PY='/data1/liuyeqiang/damot_rebuttal_20261026/env/bin/python'
HF=Path('/data1/liuyeqiang/damot_apr_complement_hf_20261028_1420')
def dump(path,obj):
    path.parent.mkdir(parents=True,exist_ok=True);tmp=path.with_suffix('.json.tmp')
    tmp.write_text(json.dumps(obj,indent=2)+'\n');tmp.replace(path)
def main():
    assert json.loads((Q/'preflight/V1_config.json').read_text())['status']=='passed'
    assert json.loads((Q/'preflight/input_storage_checks.json').read_text())['status']=='passed'
    dump(Q/'preflight/state.json',dict(status='smoke_pair_running',gpus=os.environ['CUDA_VISIBLE_DEVICES'],time=time.time()))
    timings={}
    for variant in ['HF','NoICS']:
        with socket.socket() as sock:sock.bind(('',0));port=sock.getsockname()[1]
        args=[PY,'-m','torch.distributed.launch','--nproc_per_node=4','--master_port='+str(port),'--use_env',str(Q/'scripts/smoke_pair.py'),'--variant',variant]
        started=time.time()
        with (Q/'logs'/('smoke_'+variant+'.log')).open('w') as log:
            result=subprocess.run(args,env=dict(os.environ,PYTHONDONTWRITEBYTECODE='1'),stdout=log,stderr=subprocess.STDOUT)
        assert result.returncode==0,(variant,result.returncode)
        root=Q/'preflight/smoke'/variant;rows_by_rank=[]
        for rank in range(4):
            assert json.loads((root/('complete_rank%d.json'%rank)).read_text())['total_iterations']==50
            rr=[]
            for stage,steps in enumerate([13,13,12,12],1):
                ss=json.loads((root/'audit'/('stage%d_rank%d_summary.json'%(stage,rank))).read_text())
                assert ss['calls']==steps and ss['history_domain_count']==stage-1
                assert ss['APR_active_iterations']==0 if stage==1 else ss['APR_active_iterations']>0
                assert math.isfinite(ss['APR_loss_mean']) and (ss['APR_loss_mean']==0 if stage==1 else ss['APR_loss_mean']>0)
                if variant=='NoICS':assert ss['bank_none'] and ss['bank_modules']==0 and ss['full_space_basis']
                records=list(csv.DictReader((root/'audit'/('stage%d_rank%d_iterations.csv'%(stage,rank))).open()))
                assert len(records)==steps
                if variant=='NoICS':assert all('trim' not in r['loss_keys'] and r['bank_none']=='True' for r in records)
                rr.extend(records)
            assert len(rr)==50;rows_by_rank.append(rr)
        all_times=[max(float(rows_by_rank[r][i]['iteration_seconds']) for r in range(4)) for i in range(50)]
        all_compute=[max(float(rows_by_rank[r][i]['compute_seconds']) for r in range(4)) for i in range(50)]
        steady=[i for i in range(50) if int(rows_by_rank[0][i]['iteration'])>3]
        timings[variant]=dict(critical_rank_iteration_mean_all50=sum(all_times)/50,
            critical_rank_iteration_mean_steady38=sum(all_times[i] for i in steady)/len(steady),
            critical_rank_compute_mean_steady38=sum(all_compute[i] for i in steady)/len(steady),
            wall_seconds=time.time()-started,steady_indices=steady,phase_steps=[13,13,12,12])
    log=(Q/'logs/smoke_NoICS.log').read_text()
    assert 'loss_track_trim' not in log and '[CAUSAL_PCA_' not in log
    code=json.loads((Q/'preflight/V3_code_snapshot.json').read_text())
    for item in code['files']:
        assert hashlib.sha256((Q/'code'/item['path']).read_bytes()).hexdigest()==item['sha256']
        assert hashlib.sha256((HF/'code'/item['path']).read_bytes()).hexdigest()==item['HF_sha256']
    timing_delta=100*(timings['NoICS']['critical_rank_iteration_mean_steady38']/timings['HF']['critical_rank_iteration_mean_steady38']-1)
    dump(Q/'preflight/V2_smoke.json',dict(status='passed',seed=1790355234,physical_gpus=os.environ['CUDA_VISIBLE_DEVICES'],
        iterations_per_variant=50,phase_steps=[13,13,12,12],same_cards_and_code=True,
        weights_discarded=True,formal_starts_from_Clear_S=True,NoICS_bank_module_count=0,
        NoICS_trim_loss_absent=True,APR_history_counts=[0,1,2,3],APR_stage1_zero=True,
        APR_later_stages_finite_nonzero=True,timings=timings,steady_iteration_time_change_percent=timing_delta,
        timing_rule='max rank duration per matching iteration; average excluding first3 of each disposable stage'))
    dump(Q/'preflight/validation.json',dict(status='passed',V1='passed',V2='passed',V3='passed',code_files=code['all_regular_files'],
        seed=1790355234,physical_gpus=os.environ['CUDA_VISIBLE_DEVICES'],time=time.time()))
    dump(Q/'manifests/pilot_complete.json',dict(status='complete',validation='preflight/validation.json',
        steps=50,ranks=4,formal_weights_saved=False,disposable_pair=['HF','NoICS']))
    dump(Q/'preflight/state.json',dict(status='passed',time=time.time()))
    print('ALL_PREFLIGHT_CHECKS_PASSED',timing_delta,flush=True)
if __name__=='__main__':
    try:main()
    except BaseException:
        dump(Q/'preflight/state.json',dict(status='failed',error=traceback.format_exc(),time=time.time()));raise
