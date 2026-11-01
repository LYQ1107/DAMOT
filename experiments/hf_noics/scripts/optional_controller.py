"""Use idle inference slots after each stage's required cells for optional S3."""
import csv
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time
import traceback
Q=Path(__file__).resolve().parents[1]
PY='/data1/liuyeqiang/damot_rebuttal_20261026/env/bin/python'
DOMAINS=['overcast','rainy','dawn_dusk','night']
def dump(path,value):
    path.parent.mkdir(parents=True,exist_ok=True);tmp=path.with_suffix('.json.tmp');tmp.write_text(json.dumps(value,indent=2)+'\n');tmp.replace(path)
def state(status,**extra):dump(Q/'manifests/optional_controller.json',dict(status=status,pid=os.getpid(),time=time.time(),**extra))
def csvread(p):
    with p.open() as f:return list(csv.DictReader(f))
def csvwrite(p,rows):
    with p.open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
def main():
    for stage in range(1,5):
        while not all((Q/'results'/('stage%d'%stage)/d/'complete.json').exists() for d in DOMAINS[:stage]):
            p=Q/'manifests/controller.json'
            if p.exists() and json.loads(p.read_text())['status'] in ['failed','preflight_failed_no_formal_training']:
                state('deferred_main_run_failure');return
            state('waiting_for_required_stage_cells',stage=stage);time.sleep(15)
        state('extracting_clean_features',stage=stage)
        args=[sys.executable,str(Q/'scripts/gpu_job.py'),'--name','optional_stage%d_probe'%stage,'--gpus','1',
            '--prefer','5,7,9,4,2,3,6,8','--',PY,str(Q/'scripts/optional_stage_probe.py'),'--stage',str(stage)]
        with (Q/'logs'/('optional_stage%d_probe.log'%stage)).open('w') as log:
            subprocess.run(args,stdout=log,stderr=subprocess.STDOUT,check=True)
    root=Q/'optional_drift';state('CPU_S3_analysis')
    env=dict(os.environ,CUDA_VISIBLE_DEVICES='',OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2',MKL_THREADING_LAYER='GNU',PYTHONDONTWRITEBYTECODE='1')
    with (Q/'logs/optional_S3_CPU.log').open('w') as log:
        subprocess.run([PY,str(Q/'analysis_tools/ics_probe_drift.py'),'--feat-root',str(root/'features'),
            '--model','NoICS','--out',str(root/'NoICS')],env=env,stdout=log,stderr=subprocess.STDOUT,check=True)
    baseline=Path('/data1/liuyeqiang/damot_ics_followup_20261029/results/drift_decomposition')
    for label in ['F','HF']:
        target=root/'reference'/label;target.mkdir(parents=True,exist_ok=True)
        for name in ['drift_energy.csv','cross_stage_retrieval.csv','drift_restoration.csv','complete.json','config_diff.json']:
            shutil.copy2(str(baseline/label/name),str(target/name))
    for stem in ['drift_energy','cross_stage_retrieval','drift_restoration']:
        rr=[]
        for label in ['F','HF','NoICS']:
            source=(root/'NoICS' if label=='NoICS' else root/'reference'/label)/(stem+'.csv');rr.extend(csvread(source))
        csvwrite(root/(stem+'.csv'),rr)
    energies=csvread(root/'drift_energy.csv');comparisons=[]
    index={(r['model'],r['from_stage'],r['to_stage'],r['bucket'],r['basis'],r['k_requested']):r for r in energies}
    for new in [r for r in energies if r['model']=='NoICS']:
        for label in ['HF','F']:
            old=index[label,new['from_stage'],new['to_stage'],new['bucket'],new['basis'],new['k_requested']]
            r={k:new[k] for k in ['from_stage','to_stage','bucket','basis','k_requested','k_effective']};r['comparison']='NoICS-'+label
            for field in ['mean_in_sq','mean_out_sq','mean_total_sq','mean_energy_share']:
                r['NoICS_'+field]=float(new[field]);r['reference_'+field]=float(old[field]);r['delta_'+field]=float(new[field])-float(old[field])
            comparisons.append(r)
    csvwrite(root/'paired_drift_comparison.csv',comparisons)
    dump(root/'complete.json',dict(status='complete',models=['F','HF','NoICS'],
        NoICS_bank_rows='not applicable; omitted because no bank exists',common_reference='F_stage1_e5 fit videos',
        statistics_split='test; video-name MD5 seed0',random_basis_seed=9973,no_new_training=True,
        coordinate_alignment='unchanged historical shared-coordinate convention; not proven gauge-invariant',
        metadata_alignment_asserted=True,NoICS_drift_rows=len([r for r in energies if r['model']=='NoICS'])))
    state('complete')
if __name__=='__main__':
    try:main()
    except BaseException:
        state('failed_optional_only',error=traceback.format_exc());raise
