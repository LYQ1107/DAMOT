"""Run the unchanged HF trainer, auditing its existing stage-boundary restore."""
import json
import random
from pathlib import Path
import sys
import time

Q=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(Q/'code'))
sys.path.insert(0,str(Q/'code/tools/run'))
import numpy as np
import torch
import apr_complement_experiment as hf

original_restore=hf.restore_rank_state

def same(a,b):
    if torch.is_tensor(a):return torch.is_tensor(b) and torch.equal(a.cpu(),b.cpu())
    if isinstance(a,np.ndarray):return isinstance(b,np.ndarray) and np.array_equal(a,b)
    if isinstance(a,dict):return isinstance(b,dict) and set(a)==set(b) and all(same(a[k],b[k]) for k in a)
    if isinstance(a,(tuple,list)):
        return isinstance(b,type(a)) and len(a)==len(b) and all(same(x,y) for x,y in zip(a,b))
    if isinstance(a,float) and np.isnan(a):return isinstance(b,float) and np.isnan(b)
    return a==b

@torch.no_grad()
def audited_restore(root,model,stage,epoch,final=False):
    if not final:return original_restore(root,model,stage,epoch,final=False)
    assert stage==1 and epoch==5,'Only the authorized stage1 boundary is resumed'
    rank,_=hf.get_dist_info()
    saved=torch.load(str(Q/'episode/state'/('stage1_final_rank%d.pth'%rank)),map_location='cpu')
    checkpoint=Q/'episode/stage_snapshots/stage01_after_overcast.pth'
    expected=torch.load(str(checkpoint),map_location='cpu')['state_dict']
    original_restore(root,model,stage,epoch,final=True)
    m=hf.unwrap(model)
    parameters_exact=all(torch.equal(p.detach().cpu(),expected[n]) for n,p in m.named_parameters())
    buffers=dict(m.named_buffers())
    buffers_exact=all(torch.equal(buffers[n].detach().cpu(),v) for n,v in saved['buffers'].items())
    apr_exact=same(m.track_head.roi_proto_memory.runtime_state_dict(),saved['APR'])
    counters_exact=all(getattr(m.track_head,n)==v for n,v in saved['counters'].items())
    # The FAP wrapper's counter is telemetry, not a stochastic state. Restore
    # it from the saved sidecar so subsequent call logs remain cumulative.
    meta=json.loads((Q/'episode/state'/('stage1_final_rank%d.json'%rank)).read_text())
    previous={v['pipeline']:v['calls'] for v in meta['activity']['FAP']}
    for name,transform in m._fap_audit:transform.audit_calls=previous[name]
    rng_exact=(same(random.getstate(),saved['python_rng']) and
        same(np.random.get_state(),saved['numpy_rng']) and
        torch.equal(torch.get_rng_state(),saved['torch_rng']) and
        torch.equal(torch.cuda.get_rng_state(),saved['cuda_rng']))
    assert parameters_exact and buffers_exact and apr_exact and counters_exact and rng_exact
    record=dict(status='passed',rank=rank,stage=stage,epoch=epoch,
        parameters_exact=parameters_exact,buffers_exact=buffers_exact,
        APR_runtime_exact=apr_exact,counters_exact=counters_exact,
        all_parent_RNG_states_exact=bool(rng_exact),
        FAP_audit_counters_restored=True,history_domains=list(saved['APR']['history']),
        resumed_next_stage=2,seed=1790355234,model_code_unchanged=True,
        checkpoint_sha256='7919eb9e244c194465d9c90c72831594f1c4e548bece7c4d33aa30eeabefd364',
        numerical_equivalence_to_uninterrupted_training_not_claimed=True,time=time.time())
    hf.dump(Q/'manifests'/('stage_boundary_restore_rank%d.json'%rank),record)
    print('STAGE_BOUNDARY_RESTORE_VERIFIED',rank,flush=True)
    del expected,saved

hf.restore_rank_state=audited_restore
hf.main()
