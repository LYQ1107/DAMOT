"""Run the two independent CPU analyses, then apply the fixed Stage A gate."""
import json
import os
import subprocess
import time
import traceback
from pathlib import Path

W = Path(__file__).resolve().parents[1]
PY = '/data1/liuyeqiang/damot_rebuttal_20261026/env/bin/python'
ANALYSIS = W / 'code/tools/analysis/idapr_offline.py'


def state(status, **kwargs):
    path = W / 'manifests/A_controller.json'
    value = dict(status=status, pid=os.getpid(), time=time.time(), **kwargs)
    temp = path.with_suffix('.tmp')
    temp.write_text(json.dumps(value, indent=2) + '\n')
    temp.replace(path)


def main():
    env = dict(os.environ, CUDA_VISIBLE_DEVICES='', OMP_NUM_THREADS='2',
               MKL_NUM_THREADS='2', OPENBLAS_NUM_THREADS='2',
               MKL_THREADING_LAYER='GNU', PYTHONHASHSEED='0', PYTHONUNBUFFERED='1')
    children = []
    for model in ['F', 'HF']:
        command = [PY, str(ANALYSIS), '--work-dir', str(W), '--model', model]
        log = (W / 'logs' / ('A_' + model + '.log')).open('w')
        proc = subprocess.Popen(command, env=env, stdout=log, stderr=subprocess.STDOUT)
        children.append((model, proc, log, command))
    state('running_CPU_only', children=[dict(model=m, pid=p.pid, command=c)
                                      for m, p, _, c in children])
    results = []
    for model, proc, log, _ in children:
        code = proc.wait()
        log.close()
        results.append(dict(model=model, returncode=code))
    if any(r['returncode'] for r in results):
        state('technical_failure', results=results)
        raise RuntimeError('Stage A analysis failed; inspect per-model logs')
    with (W / 'logs/A_aggregate.log').open('w') as log:
        subprocess.run([PY, str(ANALYSIS), '--work-dir', str(W), '--aggregate'],
                       env=env, stdout=log, stderr=subprocess.STDOUT, check=True)
    gate = json.loads((W / 'A/gate.json').read_text())
    state('A_passed_ready_for_U' if gate['passed'] else 'A_failed_stop_publish', gate=gate)


if __name__ == '__main__':
    try:
        main()
    except Exception:
        state('technical_failure', error=traceback.format_exc())
        raise
