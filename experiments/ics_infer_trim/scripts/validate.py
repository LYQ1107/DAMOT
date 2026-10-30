"""Validate the promised batch1 deliverables, without repeating model inference."""
from common import *
import math
assert json.loads((ROOT/'manifests/selftest.json').read_text())['status']=='PASS'
assert json.loads((ROOT/'manifests/fixed_basis_provenance.json').read_text())['status']=='PASS'
assert json.loads((ROOT/'manifests/baseline_bytes.json').read_text())['status']=='PASS'
for model in ['C0','C2']:assert json.loads((ROOT/'manifests'/('reuse_'+model+'_night.json')).read_text())['status']=='PASS'
frames=0
for m in ['C0','C2','C4']:
    for d,nf,nv in [('night',7286,36),('dawn_dusk',6476,33)]:
        r=json.loads((ROOT/'jobs'/m/d/'inference_complete.json').read_text())
        assert r['status']=='complete' and r['frames']==nf and r['videos']==nv and r['beta0_bitwise_equal_to_frozen_all_frames'];frames+=nf
        for c in cases(m):
            out=ROOT/'results'/c['label']/d;complete=json.loads((out/'metrics_complete.json').read_text())
            assert complete['aggregate_parity'] and complete['test_videos']==(25 if d=='night' else 17)
            parity=rows(out/'aggregate_parity.csv');assert len(parity)==10 and all(r['passed']=='True' for r in parity)
            agg=rows(out/'metrics.csv');assert len(agg)==7
            for row in agg:
                assert all(math.isfinite(float(row[k])) for k in ['HOTA','DetA','IDF1','AssA','IDSW'])
            per=rows(out/'per_video.csv');assert len(per)==2*nv
out=ROOT/'analysis/batch1';effect=rows(out/'paired_effects.csv');assert len(effect)==120
assert len(rows(out/'aggregate_metrics.csv'))==98
assert len(rows(out/'video_differences.csv'))==336
for r in effect:
    assert float(r['ci_low'])<=float(r['ci_high'])
    assert int(r['videos'])==dict(night=25,dawn_dusk=17,pooled_test_videos=42)[r['domain']]
assert (ROOT/'SUMMARY.md').exists()
dump(ROOT/'manifests/validation_batch1.json',dict(status='PASS',bitwise_checked_model_frames=frames,
     inference_jobs=6,aggregate_cells=98,native_aggregate_parity_checks=140,paired_effects=120,paired_video_rows=336,
     expected_test_videos=dict(night=25,dawn_dusk=17),single_training_seed=True))
print('BATCH1_DELIVERY_VALIDATION_PASS',frames,flush=True)
