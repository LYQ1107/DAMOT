"""Default GT arrays stay bitwise equal; teacher mode works with no GT boxes."""
from common import *
import subprocess
import numpy as np
import torch

reservation = torch.zeros(1, device='cuda')
annotation = RUN / 'data/annotations/dawn_dusk.json'
metadata = json.loads(annotation.read_text())
metadata['annotations'] = []
metadata_only = Q / 'configs/dawn_dusk_images_only.json'
dump(metadata_only, metadata)
common = ['--checkpoint', INPUTS['HF_stage3_snapshot'], '--config', INPUTS['HF_stage3_config'],
          '--views', 'clean,fap,photo,student', '--dose-frames-per-video', '0',
          '--max-videos', '1', '--frames-per-video', '2', '--out-dir', str(Q / 'feats')]
cases = [('gt_original', Q / 'code_original', annotation, []),
         ('gt_modified', Q / 'code_hf', annotation, []),
         ('teacher_without_GT', Q / 'code_hf', metadata_only, ['--box-source', 'teacher'])]
for name, code, ann, extra in cases:
    cmd = [sys.executable, str(code / 'tools/analysis/ics_probe_extract.py'),
           '--label', name, '--domain', 'dawn_dusk', str(ann), str(Q / 'data_views/dawn_dusk')]
    cmd += common + extra
    with (Q / 'logs' / (name + '.log')).open('w') as log:
        subprocess.run(cmd, stdout=log, stderr=subprocess.STDOUT, check=True)
def bitwise(a, b):
    return a.shape == b.shape and a.dtype == b.dtype and a.tobytes() == b.tobytes()
with np.load(str(Q / 'feats/gt_original/dawn_dusk.npz'), allow_pickle=False) as a, np.load(str(Q / 'feats/gt_modified/dawn_dusk.npz'), allow_pickle=False) as b:
    assert set(a.files) == set(b.files)
    gt_checks = {key: bitwise(a[key], b[key]) for key in a.files}
    gt_rois = len(a['video'])
    assert gt_rois > 0 and all(gt_checks.values())
with np.load(str(Q / 'feats/HF_stage3_teacher/dawn_dusk.npz'), allow_pickle=False) as full, np.load(str(Q / 'feats/teacher_without_GT/dawn_dusk.npz'), allow_pickle=False) as short:
    assert len(short['video']) > 0 and (short['instance_id'] == -1).all()
    assert (short['teacher_score'] >= .75).all()
    keep = np.isin(full['image_id'], np.unique(short['image_id']))
    teacher_checks = {}
    for key in short.files:
        reference = full[key] if key.startswith('bank_') else full[key][keep]
        teacher_checks[key] = bitwise(reference, short[key])
    assert all(teacher_checks.values()), teacher_checks
record = dict(status='passed', default_GT_box_source=True, GT_rois=gt_rois,
              all_GT_output_arrays_bitwise_equal=all(gt_checks.values()), GT_checks=gt_checks,
              teacher_from_annotation_with_zero_GT_boxes=True,
              teacher_replay_matches_full_extraction=all(teacher_checks.values()),
              teacher_checks=teacher_checks, views=['clean', 'fap', 'photo', 'student'],
              comparison='raw array bytes, including NaN bit patterns; not NPZ zip timestamps')
dump(Q / 'U/extractor_parity.json', record)
print('EXTRACTOR_PARITY_PASSED', json.dumps(record), flush=True)
