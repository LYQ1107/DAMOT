"""Recompute frozen per-video IDs; start paired statistics when brightness is ready."""
import subprocess,time
from concurrent.futures import ThreadPoolExecutor
from common import *

def tracking(model,domain):
    with (ROOT/'logs'/('tracking_'+model+'_'+domain+'.log')).open('w') as f:
        subprocess.run([PY,'-u',str(ROOT/'scripts/tracking_per_video.py'),'--model',model,'--domain',domain],env=env(True),stdout=f,stderr=subprocess.STDOUT,check=True)

def main():
    with ThreadPoolExecutor(max_workers=2) as pool:
        fs=[pool.submit(tracking,m,d) for m in ['C0','C2','C4','C1','C3'] for d in DOMAINS]
        for f in fs:f.result()
    dump(ROOT/'manifests/batch1_tracking_complete.json',dict(status='complete'))
    while not (ROOT/'manifests/brightness_complete.json').exists():time.sleep(15)
    with (ROOT/'logs/tracking_statistics.log').open('w') as f:
        subprocess.run([PY,'-u',str(ROOT/'scripts/tracking_statistics.py'),'--batch','1'],env=env(True),stdout=f,stderr=subprocess.STDOUT,check=True)
    print('TRACKING_STATISTICS_COMPLETE',flush=True)

if __name__=='__main__':main()
