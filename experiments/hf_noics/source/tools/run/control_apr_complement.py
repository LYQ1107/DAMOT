#!/usr/bin/env python3
"""Resumable controller: four-GPU training, then ten frozen evaluations."""
import argparse
import concurrent.futures
import csv
import fcntl
import io
import json
import os
from pathlib import Path
import shutil
import socket
import subprocess
import sys
import time

STAGES = ['overcast', 'rainy', 'dawn_dusk', 'night']


def dump(path, obj):
    path = Path(path)
    tmp = path.with_suffix(path.suffix+'.tmp')
    tmp.write_text(json.dumps(obj, indent=2, ensure_ascii=False))
    tmp.replace(path)


def unused_port():
    with socket.socket() as sock:
        sock.bind(('127.0.0.1',0))
        return sock.getsockname()[1]


def exclusive_gpus(gpus):
    raw = subprocess.check_output([
        'nvidia-smi', '--id='+','.join(gpus), '--query-gpu=uuid',
        '--format=csv,noheader'], text=True)
    selected = set(raw.split())
    raw = subprocess.check_output([
        'nvidia-smi', '--query-compute-apps=gpu_uuid,pid,process_name',
        '--format=csv'], text=True)
    occupied = [r for r in csv.DictReader(io.StringIO(raw))
                if r['gpu_uuid'].strip() in selected]
    if occupied:
        raise RuntimeError('Selected GPUs are occupied; no process was stopped: '+str(occupied))


def run(root, cmd, env, label, live=False):
    stamp = str(time.time_ns())
    log = root / 'logs' / (label+'_'+stamp+'.log')
    with open(log,'w') as f:
        child = subprocess.Popen(cmd, cwd=str(root/'code'), env=env,
                                 stdout=f, stderr=subprocess.STDOUT,
                                 start_new_session=True)
        record = dict(pid=child.pid, command=cmd, log=str(log),
                      gpu=env['CUDA_VISIBLE_DEVICES'], start=time.time())
        dump(root / 'manifests' / (label+'_'+stamp+'.json'), record)
        if live:
            dump(root/'controller_child.json',record)
        code = child.wait()
    record.update(exit_code=code,end=time.time())
    dump(root/'manifests'/(label+'_'+stamp+'.json'),record)
    return record


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--run-root',type=Path,required=True)
    parser.add_argument('--gpus',default='3,4,5,6')
    args=parser.parse_args()
    root=args.run_root.resolve()
    gpus=args.gpus.split(',')
    assert len(gpus)==4 and len(set(gpus))==4
    lock=open(root/'.controller.lock','a')
    fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    previous=root/'controller_child.json'
    if previous.exists():
        p=json.loads(previous.read_text())
        cmdline=Path('/proc/%d/cmdline'%p['pid'])
        if cmdline.exists() and str(root).encode() in cmdline.read_bytes():
            raise RuntimeError('Prior training process is still alive: '+str(p))
    assert (root/'manifests/pilot_complete.json').exists()
    assert shutil.disk_usage(root).free>2.5*2**30
    dump(root/'controller.json',dict(pid=os.getpid(),start=time.time(),gpus=gpus))
    env=os.environ.copy()
    env.update(OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',
               PYTHONDONTWRITEBYTECODE='1',CUDA_VISIBLE_DEVICES=','.join(gpus))
    if not (root/'episode/train_complete.json').exists():
        exclusive_gpus(gpus)
        cmd=[sys.executable,'-m','torch.distributed.launch','--nproc_per_node=4',
             '--master_port='+str(unused_port()),'--use_env',
             str(root/'code/tools/run/apr_complement_experiment.py'),
             '--run-root',str(root),'--mode','train']
        record=run(root,cmd,env,'formal_train',live=True)
        if record['exit_code']:
            dump(root/'progress.json',dict(status='training_failed',record=record))
            raise RuntimeError('Training failed; logs preserved: '+str(record))
    assert (root/'episode/train_complete.json').exists()
    exclusive_gpus(gpus)
    jobs=[(i,d) for i in range(1,5) for d in STAGES[:i]]
    dump(root/'progress.json',dict(status='evaluating',cells=10,start=time.time()))

    def worker(slot):
        records=[]
        eval_env=env.copy()
        eval_env['CUDA_VISIBLE_DEVICES']=gpus[slot]
        for stage,domain in jobs[slot::4]:
            cmd=[sys.executable,str(root/'code/tools/run/evaluate_apr_complement.py'),
                 '--run-root',str(root),'--stage',str(stage),'--domain',domain]
            records.append(run(root,cmd,eval_env,
                               'eval_stage%d_%s'%(stage,domain)))
        return records

    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
        records=[r for group in pool.map(worker,range(4)) for r in group]
    dump(root/'manifests/evaluation_processes.json',records)
    failed=[r for r in records if r['exit_code']]
    if failed:
        dump(root/'progress.json',dict(status='evaluation_failed',failed=failed))
        raise RuntimeError('Independent evaluations completed; failures need recovery')
    cmd=[sys.executable,str(root/'code/tools/run/evaluate_apr_complement.py'),
         '--run-root',str(root),'--aggregate']
    record=run(root,cmd,env,'aggregate')
    assert record['exit_code']==0,record
    dump(root/'controller_complete.json',dict(status='complete',time=time.time()))
    print('EXPERIMENT COMPLETE',root,flush=True)


if __name__=='__main__':
    main()
