import json
import os
from pathlib import Path
import subprocess
import sys
import time
Q = Path(__file__).resolve().parents[1]
PY = '/data1/liuyeqiang/damot_rebuttal_20261026/env/bin/python'
for mode in ['original','off','residual']:
    (Q/'manifests/preflight_state.json').write_text(json.dumps(dict(status='running',phase=mode,pid=os.getpid(),time=time.time())))
    args=[sys.executable,str(Q/'scripts/gpu_job.py'),'--name','batch_'+mode,'--gpus','1',
        '--prefer','2,3,5,6,4,7,8,9','--',PY,str(Q/'scripts/preflight_batch.py'),'--mode',mode]
    with (Q/'logs'/('batch_'+mode+'.log')).open('w') as f:
        result=subprocess.run(args,stdout=f,stderr=subprocess.STDOUT)
    if result.returncode:
        (Q/'manifests/preflight_state.json').write_text(json.dumps(dict(status='failed',phase=mode,exit_code=result.returncode,time=time.time())))
        raise SystemExit(result.returncode)
(Q/'manifests/preflight_state.json').write_text(json.dumps(dict(status='U1_U2_complete',time=time.time())))
