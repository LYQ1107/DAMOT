"""Execute X1/X2 lanes concurrently, then wait for four wholly idle GPUs."""
from common import *
import subprocess,time,concurrent.futures,socket,traceback
PY=str(RUN/'env/bin/python')
def refresh():subprocess.run([sys.executable,str(Q/'scripts/summarize.py')],check=True)
def alive(pid):
    p=Path('/proc')/str(pid)/'stat'
    return p.exists() and p.read_text().split()[2]!='Z'
def adopt(pid,marker,label):
    while not marker.exists():
        if not alive(pid):raise RuntimeError('Adopted job ended without result: '+label)
        time.sleep(15)
    refresh()
def job(name,args,gpus=1,prefer='2,8,9,7,3,4,5,6'):
    cmd=[sys.executable,str(Q/'scripts/gpu_job.py'),'--name',name,'--gpus',str(gpus),'--prefer',prefer,'--']+args
    with (Q/'logs'/(name+'.log')).open('w') as f:
        p=subprocess.Popen(cmd,stdout=f,stderr=subprocess.STDOUT)
        dump(Q/'manifests'/(name+'_dispatch.json'),dict(pid=p.pid,command=cmd,time=time.time()))
        rc=p.wait()
    if rc:raise RuntimeError('%s failed (exit %d); inspect log'%(name,rc))
    refresh()
def x1():
    if os.environ.get('ICS_CONTROLLER_RESUME')=='1':
        for model,domain in [('F','night'),('F','overcast'),('HF','night'),('HF','overcast')]:
            adopt(23492,Q/'X1'/model/domain/'grad_summary.json','X1 existing lane %s/%s'%(model,domain))
        return
    adopt(8508,Q/'X1/F/night/grad_summary.json','X1 F/night')
    for model,domain in [('F','overcast'),('HF','night'),('HF','overcast')]:
        job('X1_%s_%s'%(model,domain),[PY,str(Q/'scripts/grad_compare.py'),'--model',model,'--domain',domain],prefer='2,8,9,7,3,4,5,6')
def x2_f():
    if os.environ.get('ICS_CONTROLLER_RESUME')=='1':
        job('X2_F_overcast_beta0_resume',[PY,str(Q/'scripts/evaluate.py'),'--model','F','--domain','overcast','--beta','0','--check-disabled'],prefer='9,8,7,2,3,4,5,6')
    else:
        adopt(12227,Q/'X2/F/overcast/beta0/complete.json','X2 F/overcast baseline')
    for domain,beta in [('overcast',1),('night',0),('night',1)]:
        args=[PY,str(Q/'scripts/evaluate.py'),'--model','F','--domain',domain,'--beta',str(beta)]
        if beta==0:args+=['--check-disabled']
        job('X2_F_%s_beta%d'%(domain,beta),args,prefer='9,8,7,2,3,4,5,6')
def x2_hf():
    adopt(4771,Q/'feats/HF_stage4_e5/clear_holdout_45v.npz','HF clean extraction')
    subprocess.run([PY,str(Q/'scripts/basis.py'),'--model','HF'],check=True,env=dict(os.environ,OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',MKL_THREADING_LAYER='GNU'))
    for domain in ['overcast','night']:
        for beta in [0,1]:
            args=[PY,str(Q/'scripts/evaluate.py'),'--model','HF','--domain',domain,'--beta',str(beta)]
            if beta==0:args+=['--check-disabled']
            job('X2_HF_%s_beta%d'%(domain,beta),args,prefer='8,9,7,2,3,4,5,6')
def train(group):
    root=Q/'X3'/group
    with socket.socket() as s:s.bind(('',0));port=s.getsockname()[1]
    args=[PY,'-m','torch.distributed.launch','--nproc_per_node=4','--master_port='+str(port),str(Q/'code_hf/tools/run/train_seq3_episode.py'),'--base-config',str(root/'base.py'),'--source-ckpt',INPUTS['source_checkpoint'],'--target-order','overcast','rainy','dawn_dusk','night','--eval-domains','overcast','rainy','dawn_dusk','night','--work-dir',str(root/'episode'),'--epochs-per-stage','2','--resume-episode-from-stage','4','--samples-per-gpu','16','--workers-per-gpu','2','--launcher','pytorch','--no-validate','--seed','1790355234','--target-config-template',str(Q/'configs/night_target.py'),'--apr-history',str(Q/'X3/shared_apr_history.pth')]
    job(group+'_train',args,gpus=4,prefer='2,7,8,9,3,4,5,6')
    for domain in ['night','dawn_dusk']:
        job(group+'_eval_'+domain,[PY,str(Q/'scripts/evaluate.py'),'--model','HF','--domain',domain,'--checkpoint',str(root/'episode/adapt/stage04_after_night/epoch_2.pth'),'--out',str(root/'evaluation'/domain),'--beta','0'],prefer='8,9,7,2,3,4,5,6')
def main():
    dump(Q/'manifests/controller.json',dict(status='running',pid=os.getpid(),time=time.time()))
    with concurrent.futures.ThreadPoolExecutor(3) as pool:
        fs=[pool.submit(fn) for fn in (x1,x2_f,x2_hf)]
        for f in concurrent.futures.as_completed(fs):f.result()
    dump(Q/'manifests/controller.json',dict(status='X3_waiting_for_four_idle_gpus',pid=os.getpid(),time=time.time()))
    for group in ['X3-0','X3-1','X3-2','X3-3']:train(group)
    refresh();dump(Q/'manifests/controller.json',dict(status='experiments_complete_awaiting_publication',pid=os.getpid(),time=time.time()))
if __name__=='__main__':
    try:main()
    except Exception:
        dump(Q/'manifests/controller_failure.json',dict(error=traceback.format_exc(),time=time.time()));raise
