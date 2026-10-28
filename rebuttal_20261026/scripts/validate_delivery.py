#!/usr/bin/env python3
"""Acceptance checks run only after all required experiment processes finish."""
import json,csv,pickle,hashlib,time,datetime
from pathlib import Path
import numpy as np
RUN=Path(__file__).resolve().parents[1]
def read(p):
    with open(p) as f:return list(csv.DictReader(f))
def load(p):return json.loads(p.read_text())
checks=[]
expected={'exp1/apr_old_domain_metrics.csv':18,'exp1/apr_forgetting_metrics.csv':18,'exp1/stage_domain_metrics.csv':96,'exp2/clear_retention.csv':10,'exp3/teacher_pseudo_quality.csv':8,'exp3/teacher_pseudo_quality_per_class.csv':24,'exp3/teacher_pseudo_quality_per_video.csv':1008,'exp4/efficiency_summary.csv':6,'exp4/prototype_storage_by_stage.csv':5,'exp4/prototype_runtime_storage.csv':4}
for name,n in expected.items():
    p=RUN/'results'/name;rows=read(p);assert len(rows)==n,(name,len(rows),n);checks.append(dict(file=name,rows=n,bytes=p.stat().st_size,sha256=hashlib.sha256(p.read_bytes()).hexdigest()))
assert load(RUN/'manifests/teacher_predictions_audited.json')['exit_code']==0
assert len(load(RUN/'manifests/all_frame_hota_complete.json')['jobs'])==53
assert load(RUN/'manifests/efficiency_queue_exit.json')['exit_code']==0
assert (RUN/'summary.txt').stat().st_size>1000
for domain,n in [('clear_holdout_45v',9103),('night',7286)]:
    a=load(RUN/'manifests'/(domain+'_decode_validation.json'));assert a['all_decoded'] and a['frames']==n
windows=[]
for p in sorted((RUN/'results/exp4/raw_windows').glob('*.json')):
    a=load(p);r=a['summary'];assert r['measured_steps']==100 and r['warmup_steps']==20 and r['window_seconds']>=30 and a['finite_losses']
    for key in ['exclusive_GPU_before','exclusive_GPU_after']:
        s=a[key];assert s['device']=='9' and len(s['processes'])==1
    if r['group'] in ['FAP_ICS','FAP_ICS_APR'] and not a['activity']['covariance_calls']:
        assert 'inactive' in a['ICS_coverage']
    if r['group']=='FAP_ICS_APR':assert a['activity']['apr_loss_calls']>0
    windows.append(r)
assert len(windows)==12 and len(set(r['batch_pairs'] for r in windows))==1
for n in range(4):
    a=read(RUN/'results/exp4'/('prototype_runtime_history%d.csv'%n))[0]
    assert int(a['ICS_covariance_calls'])>0 and int(a['ICS_eigh_calls'])>0
    assert int(a['ICS_successful_updates'])>0
    assert (int(a['APR_loss_active_calls'])>0)==bool(n)
    assert (RUN/'results/exp4'/('profile_history%d.json'%n)).stat().st_size>1000
for m in ['S','F_stage1_e5','F_stage2_e5','F_stage3_e5','F_stage4_e5']:
    root=RUN/'results/exp2'/m;c=load(root/'complete.json');assert c['exit_code']==0 and c['frames']==9103 and c['videos']==45
    seen=[]
    for p in (root/'predictions').glob('*.pkl'):
        with open(p,'rb') as f:a=pickle.load(f)
        assert len(a['indices'])==len(a['outputs'])
        for output in a['outputs']:
            for kind in ['det_bboxes','track_bboxes']:
                assert len(output[kind])==6
                for boxes in output[kind]:assert np.isfinite(boxes).all()
        seen.extend(a['indices'])
    assert len(seen)==len(set(seen))==9103 and set(seen)==set(range(9103))
infer=load(RUN/'results/exp4/inference_complete.json');assert infer['complete_videos']==8 and infer['frames_per_window']>=1500
rows=read(RUN/'results/exp4/inference_repeats.csv');assert len(rows)==6 and all(int(r['measured_steps'])==infer['frames_per_window'] for r in rows)
now=datetime.datetime.now(datetime.timezone.utc)
out=dict(status='complete',exit_code=0,all_acceptance_checks_passed=True,files=checks,adaptation_windows=12,accuracy_unique_model_frames=45515+29144,teacher_extra_duplicate_forwards=6677,verified_at_UTC=now.isoformat(),verified_at_Asia_Shanghai=now.astimezone(datetime.timezone(datetime.timedelta(hours=8))).isoformat())
(RUN/'manifests/delivery_validation.json').write_text(json.dumps(out,indent=2))
progress=load(RUN/'progress.json');progress.update(status='complete',experiments={str(i):'complete' for i in range(1,5)},validated_at_Asia_Shanghai=out['verified_at_Asia_Shanghai'])
(RUN/'progress.json').write_text(json.dumps(progress,indent=2))
print(json.dumps(out,indent=2))
