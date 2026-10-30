"""Fit only the released nuisance basis from teacher boxes; no GT identities."""
from common import *
sys.path.insert(0, str(Q / 'code_hf/tools/analysis'))
import numpy as np
from ics_probe_analyze import normalize, principal_angles_deg
from idapr_basis import label_free_bases

job = json.loads((Q / 'manifests/teacher_extract_job.json').read_text())
assert job['status'] == 'complete'
path = Q / 'feats/HF_stage3_teacher/dawn_dusk.npz'
with np.load(str(path), allow_pickle=False) as data:
    assert (data['instance_id'] == -1).all()
    assert (data['category'] == 1).all()
    assert (data['teacher_score'] >= .75).all()
    z = normalize(data['feat_clean'].astype(float))
    views = [normalize(data['feat_' + name].astype(float)) for name in ['fap', 'photo', 'student']]
    _, E, stats = label_free_bases(z, data['image_id'], views, 16, 2)
    counts = dict(rois=len(z), images=int(len(np.unique(data['image_id']))),
                  videos_with_teacher_rois=int(len(np.unique(data['video']))),
                  minimum_teacher_score=float(data['teacher_score'].min()),
                  instance_ids='all -1', all_boxes_from_teacher=True)
out = Q / 'basis/E_n_k2.npy'
np.save(str(out), E)
reference = Path(INPUTS['A_HF3_nuisance_basis'])
angles = principal_angles_deg(E, np.load(str(reference)))
info = dict(basis=str(out), sha256=sha(out), shape=list(E.shape), source_npz=str(path),
            source_npz_sha256=sha(path), views=['clean', 'fap', 'photo', 'student'],
            all_Dawn_Dusk_videos_planned=33, planned_frames=990,
            frame_sampling='unchanged extractor defaults: first30frames per video, stride2; all33videos',
            teacher_score_thr=.75, identity_basis_discarded=True,
            stage_A_reference=str(reference), stage_A_reference_sha256=sha(reference),
            principal_angles_to_A_HF_stage3_nuisance2_deg=angles.tolist(),
            angle_median_deg=float(np.median(angles)), angle_max_deg=float(angles.max()),
            **counts, **stats)
dump(Q / 'basis/E_n_k2.json', info)
print('BASIS_READY', json.dumps(info), flush=True)
