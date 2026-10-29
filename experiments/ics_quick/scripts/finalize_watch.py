"""Validate and stage the completed run; publication itself uses the app API."""
from common import *
import time,subprocess,traceback
PY=str(RUN/'env/bin/python')
try:
    while not all((Q/'X3'/g/'evaluation'/d/'complete.json').exists() for g in ['X3-0','X3-1','X3-2','X3-3'] for d in ['night','dawn_dusk']):time.sleep(30)
    subprocess.run([PY,str(Q/'scripts/validate_completed.py')],check=True,env=dict(os.environ,OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',MKL_THREADING_LAYER='GNU'))
    subprocess.run([sys.executable,str(Q/'scripts/summarize.py')],check=True)
    subprocess.run([sys.executable,str(Q/'scripts/prepare_publication.py')],check=True)
    dump(Q/'manifests/finalize.json',dict(status='ready_for_github_publication',time=time.time()))
except Exception:
    dump(Q/'manifests/finalize.json',dict(status='failed',error=traceback.format_exc(),time=time.time()));raise
