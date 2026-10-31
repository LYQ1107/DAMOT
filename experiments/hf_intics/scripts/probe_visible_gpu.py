import os
from pathlib import Path
import subprocess
import sys
gpu=os.environ['CUDA_VISIBLE_DEVICES']
assert len(gpu.split(','))==1
subprocess.run([sys.executable,str(Path(__file__).with_name('stage_probe.py')),
    '--stage','4','--gpu',gpu],check=True)
