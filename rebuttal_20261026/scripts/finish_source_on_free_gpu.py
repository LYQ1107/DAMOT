#!/usr/bin/env python3
import os,signal,json,time
from run_accuracy_queue import RUN,run_job
records=[]
try:
    records.append(run_job(8,'evaluate_clear.py',['--model','S'],'clear_S_migrated_gpu8',[],RUN/'results/exp2/S/complete.json'))
    records.append(run_job(8,'diagnose_teacher.py',['--timepoint','e0'],'teacher_e0_migrated_gpu8',[RUN/'cache/weights/F_stage3_e5.pth',RUN/'manifests/night_decode_validation.json'],RUN/'results/exp3/e0/complete.json'))
finally:
    (RUN/'manifests/source_migration.json').write_text(json.dumps(dict(original_queue_pid=31184,original_source_pid=14351,reason='device4 acquired other users workloads; own source resumed by complete videos on now-free device8',records=records,time=time.time()),indent=2))
    os.kill(31184,signal.SIGCONT)
