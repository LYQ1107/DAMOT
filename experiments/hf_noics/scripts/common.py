"""Task-local analysis I/O; original ics_advantage tracking implementation."""
import csv,hashlib,json,os,sys,time
from pathlib import Path
from types import SimpleNamespace
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
CODE=ROOT/'code';CROOT=ROOT
CLASSES=['pedestrian','car','truck','bus','motorcycle','bicycle']
sys.path.insert(0,str(CODE))
from video_split import split_mask
probe=SimpleNamespace(split_mask=split_mask)
def dump(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    tmp=path.with_suffix(path.suffix+'.tmp');tmp.write_text(json.dumps(value,indent=2,allow_nan=False)+'\n');tmp.replace(path)
def readcsv(path):
    with Path(path).open() as f:return list(csv.DictReader(f))
def csvout(path,rows):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True);assert rows
    with path.open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)
    return h.hexdigest()
