"""Evaluate completed, immutable stage snapshots on wholly idle GPUs."""
import argparse
import concurrent.futures
import json
import os
from pathlib import Path
import time
import traceback

import formal_controller as controller

Q = Path(__file__).resolve().parents[1]
parser = argparse.ArgumentParser()
parser.add_argument('--stage', type=int, choices=[2, 3], required=True)
args = parser.parse_args()
marker = Q / 'manifests' / ('early_frozen_stage%d.json' % args.stage)

def record(status, **extra):
    controller.dump(marker, dict(status=status, stage=args.stage, pid=os.getpid(),
        time=time.time(), max_concurrent_inference_jobs=2,
        training_configuration_unchanged=True, **extra))

try:
    snapshot = json.loads((Q / 'episode/state' /
        ('stage%d_complete.json' % args.stage)).read_text())
    assert snapshot['status'] == 'complete' and snapshot['epoch'] == 5
    record('running', checkpoint_sha256=snapshot['checkpoint_sha256'])
    domains = ['overcast', 'rainy', 'dawn_dusk'][:args.stage]
    with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
        futures = [pool.submit(controller.idle_job,
            'early_eval_stage%d_%s' % (args.stage, domain),
            [controller.PY, str(Q/'code/tools/run/evaluate_apr_complement.py'),
             '--run-root', str(Q), '--stage', str(args.stage), '--domain', domain])
            for domain in domains]
        for future in concurrent.futures.as_completed(futures):
            future.result()
    record('complete', domains=domains, checkpoint_sha256=snapshot['checkpoint_sha256'],
        reused_by_final_controller=True)
except BaseException:
    record('failed', error=traceback.format_exc())
    raise
