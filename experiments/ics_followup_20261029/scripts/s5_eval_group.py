import subprocess,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];g=sys.argv[1]
for d in ['overcast','rainy','dawn_dusk','night']:
    subprocess.check_call([str(ROOT/'env/bin/python'),'-u',str(ROOT/'code/tools/run/ics_short_stage_eval.py'),'--root',str(ROOT/'results/S5'/('HF_'+g)),'--domain',d],cwd=str(ROOT/'code'))
