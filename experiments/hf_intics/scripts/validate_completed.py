import csv
import hashlib
import json
from pathlib import Path
Q=Path(__file__).resolve().parents[1]
def read(p):return json.loads(p.read_text())
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
assert read(Q/'manifests/preflight.json')['status']=='passed'
assert read(Q/'preflight/source_checks.json')['status']=='passed'
assert read(Q/'preflight/input_storage_checks.json')['status']=='passed'
assert read(Q/'preflight/NAS_storage_checks.json')['status']=='passed'
assert read(Q/'manifests/NAS_migration.json')['status']=='verified'
provenance=read(Q/'preflight/HF_source_provenance.json')
assert provenance['status']=='passed' and provenance['source_matches_NAS_original_copy']
assert sha(Q/'manifests/runtime_baseline.json')==provenance['baseline_manifest_sha256']
for item in read(Q/'preflight/config_integrity.json')['stages']:
    assert item['seed']==1790355234
    assert sha(Q/'configs'/('stage%d.py'%item['stage']))==item['INTICS_config_sha256']
for item in read(Q/'manifests/code_seal.json')['files']:
    assert sha(Q/'code'/item['path'])==item['sha256']
job=read(Q/'manifests/formal_job.json')
assert job['status']=='complete' and len(job['gpus'])==4
gate=read(Q/'episode/stage1_gate.json')
assert gate['gate_angle_field']=='first_direction_to_offline_span_deg'
assert abs(gate['gate_angle_deg']-gate['first_direction_to_offline_span_deg'])<1e-10
assert (gate['status']=='stop') == (gate['HOTA_delta']<=-.5 or gate['gate_angle_deg']>45)
clarification=read(Q/'manifests/angle_clarification.json')
assert clarification['status']=='resolved_continue' and clarification['resume_authorized']
if (Q/'manifests/stage_boundary_recovery.json').exists():
    for rank in range(4):
        recovery=read(Q/'manifests'/('stage_boundary_restore_rank%d.json'%rank))
        assert recovery['status']=='passed' and recovery['stage']==1 and recovery['epoch']==5
        assert recovery['parameters_exact'] and recovery['buffers_exact'] and recovery['APR_runtime_exact']
        assert recovery['counters_exact'] and recovery['all_parent_RNG_states_exact']
assert (Q/'probe/stage1/P1.json').exists()
for stage in [1,4]:
    p1root=Q/'probe'/('stage%d'%stage)
    if not (p1root/'P1.json').exists():continue
    p1=read(p1root/'P1.json')
    assert p1['bank_shape']==[2,256] and p1['bank_orthogonality_error']<1e-5
    assert sha(p1root/'bank_pcs.npy')==p1['bank_pcs_sha256']
    assert sha(p1root/'offline_fap_k2.npy')==p1['offline_basis_sha256']
stopped=gate['status']=='stop'
if stopped:
    assert read(Q/'episode/train_stopped.json')['completed_stages']==1
    assert not any((Q/'episode/stage_snapshots'/('stage%02d_after_%s.pth'%(i,d))).exists()
        for i,d in [(2,'rainy'),(3,'dawn_dusk'),(4,'night')])
else:
    assert read(Q/'episode/train_complete.json')['stages']==4
    assert read(Q/'probe/stage4/evidence/part1/INTICS/complete.json')['status']=='complete'
    assert (Q/'probe/stage4/P1.json').exists()
    for f in ['cross_view_retrieval.csv','deletion.csv','figures/C_INT_crops.pdf']:
        assert (Q/'probe/stage4/evidence/part1/INTICS'/f).exists()
domains=['overcast','rainy','dawn_dusk','night']
cells=0
for stage in range(1,2 if stopped else 5):
    stage_state=read(Q/'episode/state'/('stage%d_complete.json'%stage))
    assert stage_state['epoch']==5
    for rank in range(4):
        r=read(Q/'episode/state'/('stage%d_final_rank%d.json'%(stage,rank)))
        assert r['stage']==stage and r['epoch']==5 and r['final']
    for domain in domains[:stage]:
        c=Q/'results'/('stage%d'%stage)/domain
        done=read(c/'complete.json')
        baseline=read(Path('/data1/liuyeqiang/damot_apr_complement_hf_20261028_1420')/
            'results'/('stage%d'%stage)/domain/'complete.json')
        assert (done['videos'],done['frames'])==(baseline['videos'],baseline['frames'])
        assert done['status']=='complete' and done['all_frame_HOTA']
        rows=list(csv.DictReader((c/'metrics.csv').open()))
        assert all(r['checkpoint_sha256']==stage_state['checkpoint_sha256'] for r in rows)
        assert len(rows)==7 and {r['category'] for r in rows}=={
            'pedestrian','car','truck','bus','motorcycle','bicycle','OVERALL'}
        assert all(float(r['detector_score_thr'])==.05 and float(r['tracker_init_score_thr'])==.3 and
            float(r['tracker_obj_score_thr'])==.2 for r in rows)
        cells+=1
assert cells==(1 if stopped else 10)
report=dict(status='passed',stopped_by_prespecified_stage1_gate=stopped,
    complete_four_domain_run=not stopped,evaluation_cells=cells,
    seed=1790355234,gpus=4,epochs_per_stage=5,
    original_HF_code_seal_verified=True,preflight_all_passed=True,
    mandatory_stage1_gate_verified=True)
(Q/'manifests/final_validation.json').write_text(json.dumps(report,indent=2)+'\n')
print(json.dumps(report))
