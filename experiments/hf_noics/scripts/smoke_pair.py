"""Exactly 50 disposable updates across four domains, then discard all weights."""
import argparse
import copy
import gc
import os
from pathlib import Path
import sys
import torch
from mmcv import Config
from mmcv.runner import get_dist_info,init_dist
from mmdet.apis import set_random_seed
from mmtrack.apis import train_model
from mmtrack.core import setup_multi_processes
from mmtrack.utils import collect_env
import runtime_adapter as adapter
hf=adapter.hf
Q=Path(__file__).resolve().parents[1]
HF=Path('/data1/liuyeqiang/damot_apr_complement_hf_20261028_1420')
p=argparse.ArgumentParser();p.add_argument('--variant',choices=['HF','NoICS'],required=True)
p.add_argument('--local_rank','--local-rank',type=int,default=0);a=p.parse_args()
os.environ.setdefault('LOCAL_RANK',str(a.local_rank))
cfg=Config.fromfile(str(Q/'configs/stage1.py'))
setup_multi_processes(cfg);torch.backends.cudnn.benchmark=False
init_dist('pytorch',backend='nccl');rank,world=get_dist_info();assert world==4
hf.install_checkpointing();set_random_seed(1790355234)
root=Q/'preflight/smoke'/a.variant;root.mkdir(parents=True,exist_ok=True)
if a.variant=='HF':cfg=Config.fromfile(str(HF/'configs/stage1.py'))
model=hf.build_loaded(root,cfg,hf.OLD/'cache/weights/S.pth',source=True)
phase_steps=[13,13,12,12];offset=0
for stage,(domain,steps) in enumerate(zip(hf.STAGES,phase_steps),1):
    cfg=Config.fromfile(str((HF if a.variant=='HF' else Q)/'configs'/('stage%d.py'%stage)))
    # Both timing runs read precisely this task's validated aliases.
    datasets=cfg.data.train if isinstance(cfg.data.train,list) else [cfg.data.train]
    for ds in datasets:ds['img_prefix']=str(Q/'data/images'/domain)+'/'
    cfg.work_dir=str(root/('stage%d_work'%stage));cfg.checkpoint_config=None
    cfg.custom_hooks=[h for h in cfg.custom_hooks if h.get('type') not in ['APRExperimentStateHook','NoICSRuntimeAuditHook']]
    cfg.custom_hooks.append(dict(type='NoICSRuntimeAuditHook',root=str(root),stage=stage,
        expect_noics=a.variant=='NoICS',stop_after=steps,global_offset=offset,priority='LOWEST'))
    model.set_current_target_domain(domain);model.roi_proto_start_stage(domain);model.CLASSES=hf.CLASSES
    ds=hf.build_dataset(copy.deepcopy(cfg.data.train))
    try:train_model(model,[ds],cfg,distributed=True,validate=False,timestamp='noics_smoke',meta=dict(env_info=str(collect_env()),seed=1790355234))
    except adapter.SmokePhaseComplete:pass
    else:raise AssertionError('Disposable phase failed to stop at its fixed iteration')
    model.roi_proto_finalize_stage(domain)
    assert list(model.track_head.roi_proto_memory._history)==hf.STAGES[:stage]
    offset+=steps;del ds;gc.collect();torch.cuda.empty_cache();torch.distributed.barrier()
assert offset==50
hf.dump(root/('complete_rank%d.json'%rank),dict(status='complete',variant=a.variant,total_iterations=offset,
    phase_steps=phase_steps,history_domains=list(model.track_head.roi_proto_memory._history),
    formal_weights_saved=False,seed=1790355234))
torch.distributed.destroy_process_group();print('DISPOSABLE_SMOKE_COMPLETE',a.variant,rank,flush=True)
