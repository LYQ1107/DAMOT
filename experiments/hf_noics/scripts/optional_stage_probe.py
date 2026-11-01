"""Frozen canonical views for optional S3; no bank is constructed."""
import argparse
import json
import os
from pathlib import Path
import subprocess
Q=Path(__file__).resolve().parents[1]
PY='/data1/liuyeqiang/damot_rebuttal_20261026/env/bin/python'
p=argparse.ArgumentParser();p.add_argument('--stage',type=int,required=True);a=p.parse_args()
domains=['overcast','rainy','dawn_dusk','night'];assert a.stage in range(1,5)
checkpoint=Q/'episode/stage_snapshots'/('stage%02d_after_%s.pth'%(a.stage,domains[a.stage-1]));assert checkpoint.is_file()
root=Q/'optional_drift';label='NoICS_stage%d_e5'%a.stage
reference_meta=Path('/data1/liuyeqiang/damot_ics_followup_20261029/results/feats')/('HF_stage%d_e5'%a.stage)/'meta.json'
reference=json.loads(reference_meta.read_text())
views=','.join(reference['views'])
if not (root/'features'/label/'meta.json').exists():
    args=[PY,str(Q/'analysis_tools/ics_probe_extract.py'),'--checkpoint',str(checkpoint),
        '--config',str(Q/'configs'/('stage%d.py'%a.stage)),'--label',label,'--out-dir',str(root/'features'),
        '--views',views,'--dose-frames-per-video','0','--allow-uninitialized-bank']
    for d in ['clear_holdout_45v']+domains:
        args+=['--domain',d,'/data1/liuyeqiang/damot_rebuttal_20261026/data/annotations/'+d+'.json',str(Q/'data/images'/d)]
    root.mkdir(exist_ok=True)
    (root/('stage%d_extract_command.json'%a.stage)).write_text(json.dumps(dict(command=args,checkpoint_sha256=json.loads((Q/'episode/state'/('stage%d_complete.json'%a.stage)).read_text())['checkpoint_sha256'],standard_views_and_batch_match_HF=views,unused_dose_views_skipped=True),indent=2)+'\n')
    subprocess.run(args,env=dict(os.environ,PYTHONDONTWRITEBYTECODE='1'),check=True)
print('OPTIONAL_STAGE_FEATURES_COMPLETE',a.stage,flush=True)
