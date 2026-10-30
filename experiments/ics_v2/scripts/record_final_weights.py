"""Hash the five evaluated checkpoints without another model run."""
from common import *

assert json.loads((Q / 'manifests/final_validation.json').read_text())['status'] == 'passed'
rows = []
for group in ['C0', 'C1', 'C2', 'C3', 'C4']:
    checkpoint = Q / 'runs' / group / 'episode/adapt/stage04_after_night/epoch_2.pth'
    assert checkpoint.is_file() and not checkpoint.is_symlink()
    evaluations = []
    for domain in ['night', 'dawn_dusk']:
        cell = Q / 'evaluation' / group / domain
        complete = json.loads((cell / 'complete.json').read_text())
        assert complete['status'] == 'complete' and complete['all_frame_hota']
        evaluations.append(dict(domain=domain, frames=complete['frames'], videos=complete['videos'],
                                metrics_sha256=sha(cell / 'metrics.csv')))
    rows.append(dict(group=group, path=str(checkpoint), bytes=checkpoint.stat().st_size,
                     sha256=sha(checkpoint), epochs=2, seed=1790355234,
                     config_diff_sha256=sha(Q / 'runs' / group / 'config_diff.json'),
                     evaluations=evaluations))
dump(Q / 'manifests/final_model_provenance.json', dict(
     training_seed=1790355234, checkpoint_weights_published=False,
     checkpoints_retained_on_server=True, models=rows))
print('FINAL_MODEL_HASHES_RECORDED', len(rows), flush=True)
