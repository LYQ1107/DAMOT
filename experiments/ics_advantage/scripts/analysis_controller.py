"""Offline feature analysis after the complete first-batch inference queue."""
import subprocess,time
from common import *

def main():
    while not (ROOT/'manifests/batch1_inference_complete.json').exists():time.sleep(20)
    for script in ['feature_statistics.py','feature_figures.py']:
        with (ROOT/'logs'/(script[:-3]+'.log')).open('w') as f:
            subprocess.run([PY,'-u',str(ROOT/'scripts'/script),'--batch','1'],env=env(True),stdout=f,stderr=subprocess.STDOUT,check=True)
    dump(ROOT/'manifests/batch1_features_complete.json',dict(status='complete'))
    print('FEATURES_ANALYSIS_COMPLETE',flush=True)

if __name__=='__main__':main()
