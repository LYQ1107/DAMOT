"""HF files remain byte-identical; adapt only ICS-specific runner telemetry/checks."""
import ast
import copy
import csv
import inspect
import json
from pathlib import Path
import sys
import time
Q=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(Q/'code'))
sys.path.insert(0,str(Q/'code/tools/run'))
import numpy as np
import torch
from mmcv.runner import HOOKS,Hook,get_dist_info
import apr_complement_experiment as hf

class SmokePhaseComplete(Exception):pass

def activity(model):
    m=hf.unwrap(model);h=m.track_head;b=h.causal_bank
    return dict(bank_exists=b is not None,bank_initialized=False if b is None else bool(b.initialized.item()),
        ICS_num_updates=0 if b is None else int(b.num_updates.item()),
        ICS_last_effective_rois=0 if b is None else int(b.last_num_samples.item()),
        ICS_last_update_ok=False if b is None else bool(b.last_update_ok.item()),
        APR_calls=h._apr_couple_calls,APR_basis_ready_calls=h._apr_couple_basis_ready,
        APR_coupled_loss_active_calls=h._apr_couple_loss_active,APR_memory=h.roi_proto_memory.summary(),
        APR_full_space_audit=getattr(m,'_noics_stage_audit',{}),
        FAP=[dict(pipeline=n,mode=t.mode,calls=t.audit_calls) for n,t in m._fap_audit],
        allocated_MiB=torch.cuda.max_memory_allocated()/2**20,reserved_MiB=torch.cuda.max_memory_reserved()/2**20)
hf.activity=activity

@HOOKS.register_module()
class NoICSRuntimeAuditHook(Hook):
    def __init__(self,root,stage,expect_noics=True,stop_after=None,global_offset=0):
        self.root=Path(root);self.stage=int(stage);self.expect_noics=expect_noics
        self.stop_after=stop_after;self.offset=int(global_offset)
    def before_run(self,runner):
        self.rank,_=get_dist_info();self.m=hf.unwrap(runner.model);self.h=self.m.track_head
        self.calls=0;self.active=0;self.apr_sum=0.;self.total_sum=0.;self.rows=[];self.previous_end=None
        self.audit=self.root/'audit';self.audit.mkdir(parents=True,exist_ok=True)
        self.csvpath=self.audit/('stage%d_rank%d_iterations.csv'%(self.stage,self.rank))
        self.summarypath=self.audit/('stage%d_rank%d_summary.json'%(self.stage,self.rank))
        if self.stop_after:assert not self.csvpath.exists()
        self.bank_modules=sum(type(x).__name__=='CausalDirectionBank' for x in self.m.modules())
        if self.expect_noics:
            assert self.bank_modules==0 and self.h.causal_bank is None
            assert self.m.teacher_track_head.causal_bank is None
            assert not self.h.use_causal_direction_bank and not self.h.roi_proto_couple_ics
        original=getattr(self.h,'_noics_original_forward',self.h.forward_train)
        self.h._noics_original_forward=original
        self.last_losses={}
        def measured(*args,**kwargs):
            losses=original(*args,**kwargs)
            self.last_losses={k:float(v.detach()) for k,v in losses.items() if torch.is_tensor(v)}
            return losses
        self.h.forward_train=measured
        if not hasattr(self.m,'_noics_stage_audit'):self.m._noics_stage_audit={}
    def before_train_iter(self,runner):
        torch.cuda.synchronize();self.started=time.monotonic()
    def snapshot(self,status):
        history=list(self.h.roi_proto_memory._history)
        return dict(status=status,stage=self.stage,domain=hf.STAGES[self.stage-1],rank=self.rank,
            calls=self.calls,APR_active_iterations=self.active,APR_loss_mean=self.apr_sum/max(1,self.calls),
            total_loss_mean=self.total_sum/max(1,self.calls),history_domains=history,
            history_domain_count=len(history),bank_modules=self.bank_modules,
            bank_none=self.h.causal_bank is None,expect_noics=self.expect_noics,
            full_space_basis=self.h._get_apr_basis()[0] is None,
            seed=1790355234,disposable_weights=self.stop_after is not None,time=time.time())
    def save(self,status):
        record=self.snapshot(status);hf.dump(self.summarypath,record)
        self.m._noics_stage_audit[str(self.stage)]=record
        return record
    def after_train_iter(self,runner):
        torch.cuda.synchronize();now=time.monotonic();self.calls+=1
        assert np.isfinite(float(runner.outputs['loss'].detach()))
        assert all(np.isfinite(v) for v in self.last_losses.values())
        history=list(self.h.roi_proto_memory._history)
        assert history==hf.STAGES[:self.stage-1],history
        apr=self.last_losses.get('loss_roi_proto',0.)
        if self.stage==1:assert apr==0.
        if self.expect_noics:
            assert self.h.causal_bank is None and not any('trim' in k for k in self.last_losses)
            basis,beta=self.h._get_apr_basis();assert basis is None and beta==0.
            assert self.h._apr_couple_basis_ready==0 and self.h._apr_couple_loss_active==0
        self.active+=int(apr>0.);self.apr_sum+=apr;total=float(runner.outputs['loss'].detach());self.total_sum+=total
        row=dict(stage=self.stage,domain=hf.STAGES[self.stage-1],rank=self.rank,iteration=runner.iter+1,
            global_smoke_iteration=self.offset+self.calls if self.stop_after else '',
            total_loss=total,APR_loss=apr,history_domains='|'.join(history),bank_none=self.h.causal_bank is None,
            loss_keys='|'.join(sorted(self.last_losses)),compute_seconds=now-self.started,
            iteration_seconds=now-(self.previous_end or self.started))
        self.previous_end=now;self.rows.append(row)
        with self.csvpath.open('a',newline='') as f:
            writer=csv.DictWriter(f,fieldnames=list(row))
            if f.tell()==0:writer.writeheader()
            writer.writerow(row)
        if self.calls<=3 or self.calls%20==0:
            record=self.save('training')
            runner.logger.info('[%s_AUDIT] stage=%d iteration=%d bank_none=%s history_domains=%s APR_loss=%.8f APR_active=%d',
                'NOICS' if self.expect_noics else 'HF_SMOKE',self.stage,runner.iter+1,
                self.h.causal_bank is None,history,apr,self.active)
        if self.stop_after and self.calls==self.stop_after:
            if self.stage>1:assert self.active>0
            self.save('smoke_phase_complete');torch.distributed.barrier();raise SmokePhaseComplete()
    def after_train_epoch(self,runner):self.save('epoch_complete')
    def after_run(self,runner):
        if self.stage>1:assert self.active>0
        self.save('stage_training_complete')

def verify_full_space_stage(root,model,stage,rank):
    m=hf.unwrap(model);record=m._noics_stage_audit.get(str(stage))
    if record is None:
        record=json.loads((Path(root)/'audit'/('stage%d_rank%d_summary.json'%(stage,rank))).read_text())
    assert record['history_domains']==hf.STAGES[:stage-1]
    assert record['APR_active_iterations']>0 and record['APR_loss_mean']>0
    assert record['bank_none'] and record['bank_modules']==0 and record['full_space_basis']
    return True

def install_train_compatibility():
    original=ast.parse(inspect.getsource(hf.train));changed=copy.deepcopy(original);replaced=[]
    for node in ast.walk(changed):
        if not isinstance(node,ast.Assert):continue
        description=ast.dump(node.test)
        if 'causal_bank' in description and 'initialized' in description:
            replaced.append(('bank_assert',ast.dump(node)))
            node.test=ast.parse('model.track_head.causal_bank is None',mode='eval').body
            node.msg=ast.Constant(value='NoICS bank unexpectedly exists')
        elif '_apr_couple_loss_active' in description:
            replaced.append(('APR_assert',ast.dump(node)))
            node.test=ast.parse('_verify_noics_stage(root,model,stage,rank)',mode='eval').body
            node.msg=ast.Constant(value='Full-space APR did not become active')
    assert len(replaced)==2 and {x[0] for x in replaced}=={'bank_assert','APR_assert'}
    def without_asserts(tree):
        tree=copy.deepcopy(tree)
        class Remove(ast.NodeTransformer):
            def visit_Assert(self,node):return ast.Pass()
        return ast.dump(Remove().visit(tree),include_attributes=False)
    assert without_asserts(original)==without_asserts(changed)
    hf.__dict__['_verify_noics_stage']=verify_full_space_stage
    exec(compile(ast.fix_missing_locations(changed),str(Q/'scripts/runtime_adapter.py'),'exec'),hf.__dict__)
    return dict(status='passed',HF_snapshot_files_unchanged=True,
        training_AST_identical_except_two_assertions=True,changed_assertions=[x[0] for x in replaced],
        other_runtime_override='None-safe read-only activity logger',model_numerical_operations_changed=False,
        forward_observation='read-only audit wrapper returns the original loss object unchanged')
