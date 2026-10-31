"""Paired HF/new-mode timing runs; each disposable, same seed and 50 steps."""
import argparse
import copy
import csv
import json
import os
from pathlib import Path
import sys
import time
Q=Path(__file__).resolve().parents[1]
HF=Path('/data1/liuyeqiang/damot_apr_complement_hf_20261028_1420')
p=argparse.ArgumentParser()
p.add_argument('--variant',choices=['HF','INTICS'],required=True)
p.add_argument('--local_rank','--local-rank',type=int,default=0)
a=p.parse_args()
CODE=Q/('code_original' if a.variant=='HF' else 'code')
sys.path.insert(0,str(CODE/'tools/run'));sys.path.insert(0,str(CODE))
import numpy as np
import torch
from mmcv import Config
from mmcv.runner import HOOKS,Hook,get_dist_info,init_dist
from mmdet.apis import set_random_seed
from mmtrack.apis import train_model
from mmtrack.utils import collect_env
from darth.datasets import build_dataset
import apr_complement_experiment as hf
os.environ.setdefault('LOCAL_RANK',str(a.local_rank))
torch.set_num_threads(1)
torch.backends.cudnn.benchmark=False
cfg=Config.fromfile(str((HF if a.variant=='HF' else Q)/'configs/stage1.py'))
from mmtrack.core import setup_multi_processes
setup_multi_processes(cfg)
init_dist('pytorch',backend='nccl')
rank,world=get_dist_info()
assert world==4
hf.install_checkpointing()
root=Q/'preflight/smoke'/a.variant
root.mkdir(parents=True,exist_ok=True)
set_random_seed(int(cfg.seed))
cfg.work_dir=str(root/'work')
cfg.custom_hooks=[h for h in cfg.custom_hooks if h.get('type')!='APRExperimentStateHook']
cfg.custom_hooks.append(dict(type='INTICSSmokeAuditHook',priority='LOWEST'))
cfg.checkpoint_config=None
model=hf.build_loaded(root,cfg,Path(hf.OLD)/'cache/weights/S.pth',source=True)
model.set_current_target_domain('overcast')
model.roi_proto_start_stage('overcast')
model.CLASSES=hf.CLASSES

class SmokeComplete(Exception):
    pass

@HOOKS.register_module()
class INTICSSmokeAuditHook(Hook):
    def before_run(self,runner):
        self.rows=[];self.calls=[];self.first_init=None;self.iteration=0
        assert not (root/('iterations_rank%d.csv'%rank)).exists()
        self.head=hf.unwrap(runner.model).track_head
        bank=self.head.causal_bank
        trim=bank.trim;update=bank.update
        def measured_trim(feats,*args,**kwargs):
            result=trim(feats,*args,**kwargs)
            self.calls.append(not torch.equal(feats,result))
            return result
        def measured_update(feats,*args,**kwargs):
            before=bool(bank.initialized.item())
            result=update(feats,*args,**kwargs)
            if not before and bool(bank.initialized.item()) and self.first_init is None:
                self.first_init=self.iteration
            return result
        bank.trim=measured_trim;bank.update=measured_update
        forward=self.head.forward_train;self.last_losses={}
        def measured_forward(*args,**kwargs):
            losses=forward(*args,**kwargs)
            self.last_losses={k:float(v.detach()) for k,v in losses.items() if torch.is_tensor(v)}
            return losses
        self.head.forward_train=measured_forward
        self.previous_end=None
    def before_train_iter(self,runner):
        torch.cuda.synchronize()
        self.started=time.monotonic();self.iteration=runner.iter+1;self.calls=[]
        if a.variant=='INTICS' and self.iteration==1:
            detector=hf.unwrap(runner.model).detector
            assert not any(m.training for m in detector.modules()
                if isinstance(m,torch.nn.modules.batchnorm._BatchNorm)), 'Clean forward would update BN'
    def after_train_iter(self,runner):
        torch.cuda.synchronize();now=time.monotonic()
        assert len(self.calls) in (0,2)
        assert all(np.isfinite(v) for v in self.last_losses.values())
        bank=self.head.causal_bank
        if a.variant=='INTICS':
            assert bank.center is False and bank.normalize is False
            error=float((bank.pcs @ bank.pcs.T-torch.eye(2,device=bank.pcs.device)).abs().max())
            if bool(bank.initialized.item()):assert error<1e-5
        row=dict(iteration=self.iteration,key_trim_effective=self.calls[0] if self.calls else False,
            ref_trim_effective=self.calls[1] if self.calls else False,loss=float(runner.outputs['loss'].detach()),
            loss_track_trim=self.last_losses.get('loss_track_trim'),
            compute_seconds=now-self.started,
            iteration_seconds=now-(self.previous_end or self.started),
            bank_updates=int(bank.num_updates.item()))
        self.previous_end=now;self.rows.append(row)
        path=root/('iterations_rank%d.csv'%rank)
        with path.open('a',newline='') as f:
            writer=csv.DictWriter(f,fieldnames=list(row))
            if len(self.rows)==1:writer.writeheader()
            writer.writerow(row)
        if self.iteration==50:
            n=len(self.rows)
            hf.dump(root/('summary_rank%d.json'%rank),dict(
                status='complete',variant=a.variant,iterations=n,rank=rank,
                key_effective_batches=sum(r['key_trim_effective'] for r in self.rows),
                ref_effective_batches=sum(r['ref_trim_effective'] for r in self.rows),
                first_initialized_iteration=self.first_init,
                loss_track_trim_mean=float(np.mean([r['loss_track_trim'] for r in self.rows if r['loss_track_trim'] is not None])),
                compute_mean_last40=float(np.mean([r['compute_seconds'] for r in self.rows[10:]])),
                iteration_mean_last40=float(np.mean([r['iteration_seconds'] for r in self.rows[10:]])),
                bank_center=bool(getattr(bank,'center',True)),bank_normalize=bool(bank.normalize),
                disposable_weights=True,seed=int(cfg.seed),
                samples_per_gpu=16,workers_per_gpu=2))
            torch.distributed.barrier()
            raise SmokeComplete()

dataset=build_dataset(copy.deepcopy(cfg.data.train))
try:
    train_model(model,[dataset],cfg,distributed=True,validate=False,
        timestamp='intics_smoke',meta=dict(env_info=str(collect_env()),seed=int(cfg.seed)))
except SmokeComplete:
    pass
else:
    raise AssertionError('Smoke did not stop at iteration50')
torch.distributed.destroy_process_group()
print('SMOKE_COMPLETE',a.variant,rank,flush=True)
