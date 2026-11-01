"""Required training/evaluation/video evidence, immutable source, fixed decision."""
import csv
import hashlib
import json
import math
from pathlib import Path
Q=Path(__file__).resolve().parents[1]
DOMAINS=['overcast','rainy','dawn_dusk','night']
def read(path):return json.loads(path.read_text())
def csvread(path):
    with path.open() as f:return list(csv.DictReader(f))
def sha(path):
    h=hashlib.sha256()
    with path.open('rb') as f:
        for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)
    return h.hexdigest()
assert read(Q/'preflight/validation.json')['status']=='passed'
assert read(Q/'preflight/V1_config.json')['status']=='passed'
assert read(Q/'preflight/V2_smoke.json')['status']=='passed'
code=read(Q/'preflight/V3_code_snapshot.json')
assert code['status']=='passed' and code['all_regular_files']==442 and code['python_files']==346
for item in code['files']:assert sha(Q/'code'/item['path'])==item['sha256']
for stage in read(Q/'preflight/V1_config.json')['stages']:
    assert sha(Q/'configs'/('stage%d.py'%stage['stage']))==stage['config_sha256']
changes=read(Q/'config_diff.json')['model_changes'];assert len(changes)==8
for r in changes:assert r['HF'] is True and r['NoICS'] is False
job=read(Q/'manifests/formal_job.json');assert job['status']=='complete' and len(job['gpus'])==4
for rank in range(4):assert read(Q/'preflight'/('runner_compatibility_rank%d.json'%rank))['training_AST_identical_except_two_assertions']
assert read(Q/'episode/train_complete.json')['stages']==4
cells=0
for stage in range(1,5):
    endpoint=read(Q/'episode/state'/('stage%d_complete.json'%stage))
    assert endpoint['epoch']==5 and sha(Path(endpoint['checkpoint']))==endpoint['checkpoint_sha256']
    dataset=read(Q/'manifests'/('stage%d_dataset.json'%stage))
    expected_calls=5*((dataset['actual_training_samples']+63)//64)
    for rank in range(4):
        runtime=read(Q/'episode/state'/('stage%d_final_rank%d.json'%(stage,rank)))
        assert runtime['final'] and runtime['stage']==stage and runtime['epoch']==5
        a=runtime['activity'];assert not a['bank_exists'] and not a['bank_initialized'] and a['ICS_num_updates']==0
        audit=read(Q/'audit'/('stage%d_rank%d_summary.json'%(stage,rank)))
        assert audit['calls']==expected_calls and audit['bank_modules']==0 and audit['bank_none']
        assert audit['history_domains']==DOMAINS[:stage-1] and audit['full_space_basis']
        assert audit['APR_active_iterations']==0 if stage==1 else audit['APR_active_iterations']>0
        rr=csvread(Q/'audit'/('stage%d_rank%d_iterations.csv'%(stage,rank)))
        assert len(rr)==expected_calls and [int(r['iteration']) for r in rr]==list(range(1,expected_calls+1))
        assert all('trim' not in r['loss_keys'] and r['bank_none']=='True' and math.isfinite(float(r['total_loss'])) and math.isfinite(float(r['APR_loss'])) for r in rr)
    for domain in DOMAINS[:stage]:
        cell=Q/'results'/('stage%d'%stage)/domain;done=read(cell/'complete.json')
        old=read(Path('/data1/liuyeqiang/damot_apr_complement_hf_20261028_1420')/'results'/('stage%d'%stage)/domain/'complete.json')
        assert done['status']=='complete' and done['all_frame_HOTA']
        assert (done['videos'],done['frames'])==(old['videos'],old['frames'])
        rr=csvread(cell/'metrics.csv');assert len(rr)==7
        assert all(r['checkpoint_sha256']==endpoint['checkpoint_sha256'] for r in rr)
        assert all(float(r['detector_score_thr'])==.05 and float(r['tracker_init_score_thr'])==.3 and float(r['tracker_obj_score_thr'])==.2 and r['conv_bn_fusion']=='True' for r in rr)
        cells+=1
assert cells==10 and len(csvread(Q/'results/stage_domain_metrics.csv'))==70
assert len(csvread(Q/'results/paired_stage_deltas.csv'))==140
assert len(csvread(Q/'results/continual_metrics.csv'))==24
assert len(csvread(Q/'results/paired_continual_deltas.csv'))==16
for label in ['HF','NoICS']:
    for domain in DOMAINS:
        root=Q/'tracking'/label/domain
        assert read(root/'complete.json')['status']=='complete'
        assert all(r['passed']=='True' and abs(float(r['delta']))<1e-6 for r in csvread(root/'aggregate_parity.csv'))
video=read(Q/'results/video_pairs/complete.json')
assert video['test_only'] and video['bootstrap_repetitions']==10000 and video['bootstrap_seed']==0
effects=csvread(Q/'results/video_pairs/paired_summary.csv');assert len(effects)==30
assert all(int(r['bootstrap_valid_replicates'])==10000 and float(r['ci_low'])<=float(r['ci_high']) for r in effects)
decision=read(Q/'results/decision.json');assert decision['status']=='complete'
assert decision['equivalent_by_prespecified_rule']==(decision['FinalMean_delta']>=-.3 and decision['AvgF_delta']<=.3)
assert all(r['test']=='True' for r in csvread(Q/'results/video_pairs/video_differences.csv'))
result=dict(status='passed',evaluation_cells=cells,metric_rows=70,source_files_verified=442,
    python_source_files=346,all_config_model_changes_are_the_two_requested_flags=True,
    seed=1790355234,gpus=4,epochs_per_stage=5,video_pair_protocol='ics_advantage',bootstrap_repetitions=10000,
    prespecified_decision_verified=True,equivalent_by_prespecified_rule=decision['equivalent_by_prespecified_rule'])
(Q/'manifests/final_validation.json').write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps(result))
