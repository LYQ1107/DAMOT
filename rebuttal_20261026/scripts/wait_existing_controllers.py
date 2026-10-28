#!/usr/bin/env python3
import os,time
from pathlib import Path
RUN=Path(__file__).resolve().parents[1]
names={str(RUN/'scripts'/n) for n in ['run_accuracy_queue.py','run_efficiency_queue.py','finish_source_on_free_gpu.py']}
running=[]
for p in Path('/proc').iterdir():
    if not p.name.isdigit() or int(p.name)==os.getpid():continue
    try:
        args=(p/'cmdline').read_bytes().split(b'\0')
        if any(n.encode() in args for n in names):running.append((int(p.name),p,args))
    except (FileNotFoundError,PermissionError,ProcessLookupError):pass
for pid,p,args in running:
    print('Waiting for existing task controller',pid,flush=True)
    while True:
        try:
            if (p/'cmdline').read_bytes().split(b'\0')!=args:break
        except (FileNotFoundError,PermissionError,ProcessLookupError):break
        time.sleep(20)
