import json,os,subprocess,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];group=sys.argv[1];gpu=sys.argv[2]
out=ROOT/'results/S5'/('HF_'+group);label='S5_HF_'+group
done=json.loads((out/'train_complete.json').read_text())
cmd=[str(ROOT/'env/bin/python'),'-u',str(ROOT/'code/tools/analysis/ics_probe_extract.py'),'--checkpoint',done['checkpoint'],'--config',str(out/'config.py'),'--label',label,'--out-dir',str(ROOT/'results/feats'),'--views','clean,fap','--dose-frames-per-video','0']
old=Path('/data1/liuyeqiang/damot_rebuttal_20261026')
for d in ['clear_holdout_45v','overcast','rainy','dawn_dusk','night']:cmd+=['--domain',d,str(old/'data/annotations'/(d+'.json')),str(old/'data/views'/d)]
env=dict(os.environ,CUDA_VISIBLE_DEVICES=gpu)
if not (ROOT/'results/feats'/label/'meta.json').exists():subprocess.check_call(cmd,env=env,cwd=str(ROOT/'code'))
