"""Resume completed work; require exact F parity or an explicit user decision."""
from common import *
from controller import job,train,refresh,alive
import time,subprocess,concurrent.futures,traceback,shutil
PY=str(RUN/'env/bin/python')
def state(status,**kwargs):dump(Q/'manifests/controller_v2.json',dict(status=status,pid=os.getpid(),time=time.time(),**kwargs))
def marker_wait(path,description):
    while not path.exists():
        state('waiting_for_existing_job',description=description)
        time.sleep(15)
def f_policy():
    path=Q/'manifests/F_baseline_policy.json'
    return json.loads(path.read_text()).get('mode') if path.exists() else 'require_exact_historical_parity'
def accept_current(cell):
    assert f_policy()=='use_current_environment'
    assert (cell/'metrics.csv').exists() and (cell/'baseline_parity.csv').exists()
    rows=list(csv.DictReader((cell/'metrics.csv').open()))
    assert len(rows)==7 and {r['category'] for r in rows}==set(CLASSES)|{'OVERALL'}
    assert {int(r['frames']) for r in rows}<={4448,7286}
    approval=json.loads((Q/'manifests/F_baseline_policy.json').read_text())
    dump(cell/'complete.json',dict(status='complete',baseline_parity=False,baseline='current-environment paired beta0; archived differences retained',user_approval=approval,frames=int(rows[0]['frames']),videos=int(rows[0]['videos']),all_frame_hota=True,beta=0,time=time.time()))
def f_lane():
    corrected=Q/'X2/F/overcast/beta0_cudnn_true'
    job_status=Q/'manifests/X2_F_overcast_cudnn_true_job.json'
    while True:
        r=json.loads(job_status.read_text())
        if r['status'] in ('complete','failed'):break
        time.sleep(15)
    if not (corrected/'complete.json').exists():
        assert (corrected/'baseline_parity.csv').exists(),'F correction failed before baseline metrics; independent technical recovery required'
        while f_policy()=='require_exact_historical_parity':
            state('F_baseline_needs_user_decision',parity_csv=str(corrected/'baseline_parity.csv'))
            time.sleep(15)
        if f_policy()=='skip_F_interventions':
            dump(Q/'manifests/F_interventions_skipped.json',dict(user_policy=json.loads((Q/'manifests/F_baseline_policy.json').read_text()),reason='Historical frozen F baseline could not be reproduced exactly; user elected strict skip',time=time.time()));return
        accept_current(corrected)
    state('X2_F_running',baseline_policy=f_policy())
    # Retain the first diagnostic; promote the corrected run to the canonical
    # beta0 directory only after evaluation has finished and been accepted.
    canonical=Q/'X2/F/overcast/beta0'
    diagnostic=Q/'X2/F/overcast/beta0_first_cudnn_false'
    if not diagnostic.exists():canonical.rename(diagnostic)
    if corrected.exists() and not canonical.exists():corrected.rename(canonical)
    for domain,beta in [('overcast',1),('night',0),('night',1)]:
        cell=Q/'X2/F'/domain/('beta%d'%beta)
        if (cell/'complete.json').exists():continue
        args=[PY,str(Q/'scripts/evaluate.py'),'--model','F','--domain',domain,'--beta',str(beta)]
        if beta==0:args+=['--check-disabled']
        try:job('X2_F_%s_beta%d_final'%(domain,beta),args,prefer='8,9,2,7,3,4,5,6')
        except RuntimeError:
            if beta==0 and (cell/'baseline_parity.csv').exists() and f_policy()=='use_current_environment':accept_current(cell)
            else:raise
def hf_lane():
    for domain in ['overcast','night']:
        for beta in [0,1]:
            marker=Q/'X2/HF'/domain/('beta%d'%beta)/'complete.json'
            while not marker.exists():
                if not alive(5910):
                    args=[PY,str(Q/'scripts/evaluate.py'),'--model','HF','--domain',domain,'--beta',str(beta)]
                    if beta==0:args+=['--check-disabled']
                    job('X2_HF_%s_beta%d_retry'%(domain,beta),args,prefer='8,9,2,7,3,4,5,6')
                time.sleep(15)
def main():
    state('resuming_validated_work')
    for model in ['F','HF']:
        for domain in ['night','overcast']:marker_wait(Q/'X1'/model/domain/'grad_summary.json','X1 '+model+'/'+domain)
    with concurrent.futures.ThreadPoolExecutor(2) as pool:
        for f in concurrent.futures.as_completed([pool.submit(f_lane),pool.submit(hf_lane)]):f.result()
    state('X3_waiting_for_four_wholly_idle_gpus')
    for name in ['X3-0','X3-1','X3-2','X3-3']:
        if all((Q/'X3'/name/'evaluation'/d/'complete.json').exists() for d in ['night','dawn_dusk']):continue
        state('X3_group_pending',group=name)
        train(name)
    subprocess.run([PY,str(Q/'scripts/validate_completed.py')],check=True)
    refresh();state('experiments_complete_awaiting_publication')
if __name__=='__main__':
    try:main()
    except Exception:
        state('failed',error=traceback.format_exc());raise
