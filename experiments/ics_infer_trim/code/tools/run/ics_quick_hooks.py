"""Audit only: write X3 loss activation and bank counters inside this run."""
import json, os
from pathlib import Path
from mmcv.runner import HOOKS, Hook, get_dist_info
from mmcv.parallel import is_module_wrapper
import numpy as np
import torch

@HOOKS.register_module()
class ICSQuickAuditHook(Hook):
    def __init__(self, root):
        self.root=Path(root);self.total=0;self.triggered=0
    def before_run(self,runner):
        self.model=runner.model.module if is_module_wrapper(runner.model) else runner.model
        self.head=self.model.track_head
        orig=self.head.forward_train
        def call(*args,**kwargs):
            losses=orig(*args,**kwargs);self.total+=1;self.triggered+=int('loss_track_trim' in losses)
            return losses
        self.head.forward_train=call
        self.initial_updates=None if self.head.causal_bank is None else int(self.head.causal_bank.num_updates.item())
        self.save('start',runner)
    def save(self,tag,runner):
        rank,_=get_dist_info();h=self.head;b=h.causal_bank
        s=h.roi_proto_memory.summary()
        record=dict(tag=tag,rank=rank,epoch=runner.epoch,iteration=runner.iter,history_domains=len(s['history_domains']),history_domain_names=list(h.roi_proto_memory._history),num_updates=None if b is None else int(b.num_updates.item()),num_updates_start=self.initial_updates,initialized=None if b is None else bool(b.initialized.item()),total_batches=self.total,trim_triggered_batches=self.triggered,trim_trigger_ratio=self.triggered/self.total if self.total else None,causal_freeze_random_basis=h.causal_freeze_random_basis)
        out=self.root/('audit_%s_rank%d.json'%(tag,rank));out.parent.mkdir(parents=True,exist_ok=True);out.write_text(json.dumps(record,indent=2))
        runner.logger.info('[ICS_QUICK] %s',record)
        if h.causal_freeze_random_basis:
            assert int(b.num_updates.item())==self.initial_updates,'Frozen bank was updated'
    def after_train_epoch(self,runner):self.save('epoch%d'%(runner.epoch+1),runner)
    def after_train_iter(self,runner):
        if (runner.iter+1)%20==0:self.save('progress',runner)
    def after_run(self,runner):self.save('end',runner)
