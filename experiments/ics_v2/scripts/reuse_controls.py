"""Reuse X3-0/1 metrics and IDSW, without retraining or inference."""
from common import *
import shutil

old = Path('/data1/liuyeqiang/ics_quick_20261029_160807/X3')
for group in ['X3-0', 'X3-1']:
    for domain in ['night', 'dawn_dusk']:
        source = old / group / 'evaluation' / domain
        target = Q / 'evaluation' / group / domain
        target.mkdir(parents=True, exist_ok=True)
        copied = []
        for name in ['complete.json', 'metrics.csv', 'summary_metrics.csv', 'track_metrics.csv',
                     'hota_metrics.csv', 'used_config.py', 'runtime.json']:
            if (source / name).exists():
                shutil.copy2(str(source / name), str(target / name))
                copied.append(dict(file=name, sha256=sha(source / name)))
        metrics = list(csv.DictReader((target / 'metrics.csv').open()))
        tracks = {r['']: r for r in csv.DictReader((target / 'track_metrics.csv').open())}
        for row in metrics:
            row['IDSW'] = int(float(tracks[row['category']]['IDSw']))
        write_csv(target / 'metrics.csv', metrics)
        dump(target / 'reused_from.json', dict(source=str(source), original_files=copied,
             original_metrics_sha256=sha(source / 'metrics.csv'),
             change='IDSW column added from original track_metrics.csv; four metrics unchanged',
             training_or_inference_repeated=False))
print('FOUR_CONTROL_CELLS_REUSED', flush=True)
