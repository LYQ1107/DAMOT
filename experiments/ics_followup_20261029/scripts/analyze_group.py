"""S1/S2/S3 CPU analyses; no extraction or training."""
import json,os,subprocess,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'code/tools/run'))
from apr_complement_experiment import Config
from protocol import write_views,write_json
name=sys.argv[1];py=str(ROOT/'env/bin/python');label=name+'_stage4_e5'
from check_roi_alignment import check
check(name)
source=Path('/data1/liuyeqiang/damot_apr_complement_hf_20261028_1420' if name=='HF' else '/data1/liuyeqiang/damot_apr_complement_hf_nogeom_20261029_1443')
base=[py,'-u',str(ROOT/'code/tools/analysis/ics_probe_analyze.py'),'--feat-root',str(ROOT/'results/feats'),'--label',label,'--domains','clear_holdout_45v','overcast','rainy','dawn_dusk','night','--ref-domain','clear_holdout_45v','--stage-labels']+[name+'_stage%d_e5'%k for k in range(1,5)]+['--stage-domains','overcast','rainy','dawn_dusk','night','--classes','1','--montage']
env=dict(os.environ,CUDA_VISIBLE_DEVICES='',OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2',MKL_THREADING_LAYER='GNU')
drift_out=ROOT/'results/drift_decomposition'/name
drift_cmd=[py,'-u',str(ROOT/'code/tools/analysis/ics_probe_drift.py'),'--feat-root',str(ROOT/'results/feats'),'--model',name,'--out',str(drift_out)]
drift_log=(ROOT/'logs'/('S3_'+name+'.log')).open('a');drift=subprocess.Popen(drift_cmd,env=env,stdout=drift_log,stderr=subprocess.STDOUT,cwd=str(ROOT/'code'))
for version in ['v2','v1']:
    out=ROOT/'results'/('analysis_'+label+('_v2' if version=='v2' else ''))
    if (out/'summary.json').exists() and (out/'figure_manifest.json').exists():continue
    out.mkdir(parents=True,exist_ok=True)
    write_views(Config.fromfile(str(source/'configs/stage4.py')),out/'resolved_views.txt',dict(features='same fixed-geometry probe as F; not an invocation of training view pipelines'))
    write_json(out/'config_diff.json',dict(model_changes=[],analysis_version=version,covariance_vs_bank_pcs='separate families per user approval; actual angles retained',v2_changes=['k scan 2,4,8,16,32; beta 0,.5,1; random10','Fisher on raw projected features','night binary probes and Full confusion'],original_v1_preserved=True,seed=0))
    cmd=base+['--out',str(out)]+(['--v2'] if version=='v2' else [])
    if version=='v1':cmd[2]=str(ROOT/'scripts/ics_probe_analyze_v1.py')
    with (ROOT/'logs'/('S2_'+name+'_'+version+'.log')).open('a') as log:subprocess.check_call(cmd,env=env,cwd=str(ROOT/'code'),stdout=log,stderr=subprocess.STDOUT)
assert drift.wait()==0;drift_log.close()
write_views(Config.fromfile(str(source/'configs/stage4.py')),drift_out/'resolved_views.txt',dict(basis='F_stage1_e5 fit videos; current stored bank only for ICS2'))
write_json(ROOT/'manifests'/('analysis_'+name+'_complete.json'),dict(status='complete',S1=True,S2=True,S3=True))
