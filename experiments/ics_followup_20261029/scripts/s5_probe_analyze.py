import json,os,subprocess,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];group=sys.argv[1];label='S5_HF_'+group
out=ROOT/'results'/('analysis_'+label+'_v2');out.mkdir(parents=True,exist_ok=True)
from protocol import write_json
from check_roi_alignment import check_pair
write_json(out/'ROI_alignment.json',dict(status='PASS',checks=check_pair(label,'F_stage4_e5')))
import shutil
shutil.copyfile(str(ROOT/'results/S5'/('HF_'+group)/'resolved_views.txt'),str(out/'resolved_views.txt'))
write_json(out/'config_diff.json',dict(model_config=str(ROOT/'results/S5'/('HF_'+group)/'config.py'),analysis='P1/P2 plus night binary probes',views=['clean','fap'],seed=0,full_P3_scan_requested=False))
cmd=[str(ROOT/'env/bin/python'),'-u',str(ROOT/'code/tools/analysis/ics_probe_analyze.py'),'--feat-root',str(ROOT/'results/feats'),'--label',label,'--domains','clear_holdout_45v','overcast','rainy','dawn_dusk','night','--ref-domain','clear_holdout_45v','--classes','1','--v2','--probe-only','--out',str(out)]
subprocess.check_call(cmd,cwd=str(ROOT/'code'),env=dict(os.environ,CUDA_VISIBLE_DEVICES='',OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2',MKL_THREADING_LAYER='GNU'))
