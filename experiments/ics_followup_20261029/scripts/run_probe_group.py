import os,sys,json,subprocess
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
OLD=Path('/data1/liuyeqiang/damot_rebuttal_20261026')
DOMAINS=['clear_holdout_45v','overcast','rainy','dawn_dusk','night']
STAGES=['overcast','rainy','dawn_dusk','night']
name,source,gpu=sys.argv[1:4];source=Path(source)
env=dict(os.environ,CUDA_VISIBLE_DEVICES=gpu,OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',MKL_THREADING_LAYER='GNU',PYTHONUNBUFFERED='1')
for k in [4,1,2,3]:
 label='%s_stage%d_e5'%(name,k);out=ROOT/'results/feats'/label
 if (out/'meta.json').exists():continue
 ck=source/'episode/adapt'/('stage%02d_after_%s'%(k,STAGES[k-1]))/'epoch_5.pth'
 assert ck.exists(),ck
 cmd=[str(OLD/'env/bin/python'),'tools/analysis/ics_probe_extract.py','--checkpoint',str(ck),'--config',str(source/'configs'/('stage%d.py'%k)),'--label',label,'--out-dir',str(ROOT/'results/feats'),'--views','clean,fap,photo,student' if k==4 else 'clean,fap','--dose-frames-per-video','3' if k==4 else '0']
 for d in DOMAINS:cmd+=['--domain',d,str(OLD/'data/annotations'/(d+'.json')),str(OLD/'data/views'/d)]
 job=ROOT/'manifests'/('probe_'+label+'.json');job.write_text(json.dumps(dict(command=cmd,checkpoint=str(ck),source=str(source),gpu=gpu),indent=2))
 with (ROOT/'logs'/('extract_'+label+'.log')).open('w') as log:
  subprocess.check_call(cmd,cwd=str(ROOT/'code'),env=env,stdout=log,stderr=subprocess.STDOUT)
 print('COMPLETE',label,flush=True)
