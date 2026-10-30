"""Finish CPU verification and freeze the first batch for GitHub review."""
import subprocess,time
from common import *
def run(name):
    subprocess.run([PY,'-u',str(ROOT/'scripts'/name)],env=environment(),check=True)
while not all((ROOT/'jobs'/m/d/'inference_complete.json').exists() for m in ['C0','C2','C4'] for d in DOMAINS):time.sleep(5)
run('verify_baseline_bytes.py')
while not (ROOT/'analysis/batch1/complete.json').exists():time.sleep(5)
run('refresh_case_configs.py');run('validate.py');run('build_delivery.py')
dump(ROOT/'manifests/ready_for_publication.json',dict(status='ready_for_review_and_publication',batch=1,time=time.time()))
print('BATCH1_FROZEN_FOR_PUBLICATION',flush=True)
