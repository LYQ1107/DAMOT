"""Reclaim only this run's redundant weights, after both evals are complete."""
from common import *
import time
record=[];done=set()
while len(done)<4:
    for name in ['X3-0','X3-1','X3-2','X3-3']:
        if name in done:continue
        root=Q/'X3'/name
        if not all((root/'evaluation'/d/'complete.json').exists() for d in ['night','dawn_dusk']):continue
        for rel in ['episode/stage_snapshots/stage00_source_only.pth','episode/adapt/stage04_after_night/epoch_1.pth']:
            path=root/rel
            if path.is_file() and not path.is_symlink():
                record.append(dict(path=str(path),bytes=path.stat().st_size,sha256=sha(path),reason='Redundant generated weight; group fully trained and both domain evaluations completed; epoch2 and endpoint retained'))
                path.unlink()
        done.add(name);dump(Q/'manifests/generated_weight_cleanup.json',record)
    time.sleep(30)
