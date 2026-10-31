import csv
import json
import shutil
from pathlib import Path
Q=Path(__file__).resolve().parents[1]
out=dict(Q=str(Q),free_GiB=round(shutil.disk_usage(str(Q)).free/2**30,2))
for name in ['preflight_state','smoke_controller','preflight','formal_controller']:
    p=Q/'manifests'/(name+'.json')
    if p.exists():out[name]=json.loads(p.read_text())
smoke=[]
for variant in ['HF','INTICS']:
    rr=[]
    for rank in range(4):
        p=Q/'preflight/smoke'/variant/('iterations_rank%d.csv'%rank)
        rr.append(list(csv.DictReader(p.open())) if p.exists() else [])
    n=min(map(len,rr))
    if n:
        smoke.append(dict(variant=variant,steps=n,
            key_ratio=sum(r['key_trim_effective']=='True' for rows in rr for r in rows[:n])/(4*n),
            ref_ratio=sum(r['ref_trim_effective']=='True' for rows in rr for r in rows[:n])/(4*n),
            latest_loss=[float(rows[n-1]['loss']) for rows in rr]))
out['smoke']=smoke
jobs=[]
for p in sorted((Q/'manifests').glob('*_job.json')):
    j=json.loads(p.read_text())
    if j['status'] not in ['complete','failed']:
        jobs.append(dict(name=p.stem,status=j['status'],available=j.get('available'),gpus=j.get('gpus')))
out['jobs']=jobs
out['evaluation_complete_cells']=[]
for stage in range(1,5):
    for domain in ['overcast','rainy','dawn_dusk','night'][:stage]:
        p=Q/'results'/('stage%d'%stage)/domain/'complete.json'
        if p.exists() and json.loads(p.read_text())['status']=='complete':
            out['evaluation_complete_cells'].append([stage,domain])
final_probe=Q/'probe/stage4/evidence/part1/INTICS/progress.json'
out['final_probe_phase']=(json.loads(final_probe.read_text()).get('phase')
    if final_probe.exists() else 'feature extraction pending or running')
extract_log=Q/'probe/stage4/extract.log'
if extract_log.exists():
    with extract_log.open('rb') as f:
        f.seek(0,2);f.seek(max(0,f.tell()-2048))
        tail=f.read().decode('utf8',errors='replace').splitlines()
        out['final_probe_extract_tail']=tail[-1] if tail else ''
for name in ['progress.json','episode/stage1_gate_waiting.json','episode/stage1_gate.json','episode/train_complete.json','episode/train_stopped.json']:
    p=Q/name
    if p.exists():out[name]=json.loads(p.read_text())
checks={}
for name in ['gate_stage1_overcast','gate_stage1_probe']:
    p=Q/'manifests'/(name+'_inference.json')
    if p.exists():
        j=json.loads(p.read_text());checks[name]={k:j[k] for k in ['status','gpu','pid']}
for name,relative in [('overcast_progress','logs/gate_stage1_overcast.log'),
                      ('probe_progress','probe/stage1/extract.log')]:
    p=Q/relative
    if p.exists():
        with p.open('rb') as f:
            f.seek(0,2);size=f.tell();f.seek(max(0,size-4096))
            checks[name]=f.read().decode('utf8',errors='replace').splitlines()[-2:]
cell=Q/'results/stage1/overcast'
if (cell/'complete.json').exists():
    rows=list(csv.DictReader((cell/'metrics.csv').open()))
    checks['Overcast_metrics']=[{k:r[k] for k in ['category','HOTA_percent','DetA_percent','IDF1_percent','AssA_percent']}
        for r in rows if r['category'] in ['OVERALL','car']]
p=Q/'probe/stage1/P1.json'
if p.exists():
    try:
        j=json.loads(p.read_text())
        checks['P1']={k:j[k] for k in ['first_direction_angle_deg','first_direction_to_offline_span_deg',
            'principal_angles_deg','P1_weather_shift_energy_share','P1_identity_energy_share']}
    except json.JSONDecodeError:checks['P1']='writing'
if checks:out['frozen_checks']=checks
print(json.dumps(out))
