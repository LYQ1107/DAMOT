#!/usr/bin/env python3
"""Write final provenance hashes after every experiment passes validation."""
import hashlib, json, platform, subprocess, time
from pathlib import Path

RUN=Path(__file__).resolve().parents[1]
roots=[RUN/'code',RUN/'configs',RUN/'scripts']
files=[]
for root in roots:
    if not root.exists():
        continue
    for p in sorted(root.rglob('*')):
        if p.is_file() and '__pycache__' not in p.parts and p.suffix in {'.py','.json','.txt','.sh'}:
            h=hashlib.sha256(p.read_bytes()).hexdigest()
            files.append(dict(path=str(p.relative_to(RUN)),sha256=h,bytes=p.stat().st_size))
env={}
for cmd,key in [([str(RUN/'env/bin/python'),'--version'],'python'),(['nvidia-smi','--query-gpu=index,name,driver_version','--format=csv,noheader'],'gpu')]:
    try:
        env[key]=subprocess.check_output(cmd,stderr=subprocess.STDOUT,text=True).strip()
    except Exception as e:
        env[key]=repr(e)
out=dict(run_root=str(RUN),created_UTC=time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()),host=platform.node(),code_files=files,environment=env,validated_by='manifests/delivery_validation.json',accuracy_models=['S','F_stage1_e5','F_stage2_e5','F_stage3_e5','F_stage4_e5'],teacher_points=['e0','e1','e3','e5'],efficiency_windows=12,module_histories=[0,1,2,3],deployment_repeats=3)
(RUN/'manifests/code_hashes_final.json').write_text(json.dumps(out,indent=2,ensure_ascii=False))
print(json.dumps({'code_files':len(files),'manifest':str(RUN/'manifests/code_hashes_final.json')},ensure_ascii=False))
