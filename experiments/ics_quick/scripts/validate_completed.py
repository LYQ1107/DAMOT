"""Required checks on final experiment artifacts; no additional GPU runs."""
from common import *
setup('HF')
import torch,numpy as np
records=[]
for model in ['F','HF']:
    for domain in ['night','overcast']:
        root=Q/'X1'/model/domain
        rows=list(csv.DictReader((root/'grad_compare.csv').open()))
        assert len(rows)==20 and [int(r['batch']) for r in rows]==list(range(20))
        assert all(float(r['trim_alpha'])==.1 and int(r['num_pcs'])==2 for r in rows)
        records.append(dict(kind='X1',model=model,domain=domain,batches=20,passed=True))
for model in ['F','HF']:
    for domain in ['night','overcast']:
        for beta in [0,1]:
            root=Q/'X2'/model/domain/('beta%d'%beta)
            done=json.loads((root/'complete.json').read_text())
            expected=(36,7286) if domain=='night' else (22,4448)
            assert (done['videos'],done['frames'])==expected
            if beta==0 and model=='HF':assert done['baseline_parity'] is True
            if beta==0 and model=='F' and not done.get('baseline_parity'):
                assert json.loads((Q/'manifests/F_baseline_policy.json').read_text())['mode']=='use_current_environment'
            records.append(dict(kind='X2',model=model,domain=domain,beta=beta,passed=True))
history=json.loads((Q/'X3/apr_history_manifest.json').read_text())
assert history['shared_history_sha256']==sha(Q/'X3/shared_apr_history.pth')
for group in ['X3-0','X3-1','X3-2','X3-3']:
    root=Q/'X3'/group
    cfg=json.loads((root/'resolved_causal_roi_proto.json').read_text())
    assert cfg['roi_proto_couple_ics'] is False
    assert json.loads((root/'config_diff.json').read_text())['shared_apr_history_sha256']==history['shared_history_sha256']
    for rank in range(4):
        start=json.loads((root/('audit_start_rank%d.json'%rank)).read_text());end=json.loads((root/('audit_end_rank%d.json'%rank)).read_text())
        assert start['history_domains']==end['history_domains']==3
        assert end['epoch']==2 and end['iteration']==228 and end['total_batches']==228
        if group=='X3-1':
            assert start['num_updates'] is None and end['num_updates'] is None
            assert end['trim_triggered_batches']==0
        if group=='X3-3':assert start['num_updates']==end['num_updates']
        if group in ['X3-2','X3-3']:assert end['total_batches']>0 and end['trim_triggered_batches']>0
    if group=='X3-3':
        with (root/'episode/random_basis.npy').open('rb') as f:basis=np.load(f)
        assert np.array_equal(basis,np.load(str(root/'expected_random_basis.npy')))
        ck=torch.load(str(root/'episode/adapt/stage04_after_night/epoch_2.pth'),map_location='cpu')
        assert np.array_equal(ck['state_dict']['track_head.causal_bank.pcs'].numpy().T,basis)
        del ck
    for domain in ['night','dawn_dusk']:
        done=json.loads((root/'evaluation'/domain/'complete.json').read_text())
        assert done['status']=='complete'
        assert (done['videos'],done['frames'])==((36,7286) if domain=='night' else (33,6476))
    records.append(dict(kind='X3',group=group,history_domains_start=3,epochs=2,batches_per_rank=228,four_rank_audits=True,passed=True))
dump(Q/'manifests/final_validation.json',dict(status='passed',checks=records))
print('FINAL_VALIDATION_PASSED',len(records),flush=True)
