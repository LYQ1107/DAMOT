"""Record inference/analysis versions without initializing a CUDA context."""
import platform
import subprocess
import sys
import pkg_resources
from evidence_cpu import ROOT,dump


packages={}
for name in ['torch','torchvision','mmcv','mmcv-full','mmdet','mmtrack','numpy','scipy','scikit-learn','matplotlib','Pillow','opencv-python']:
    try:packages[name]=pkg_resources.get_distribution(name).version
    except pkg_resources.DistributionNotFound:pass
dump(ROOT/'manifests/environment.json',dict(python=platform.python_version(),executable=sys.executable,
     platform=platform.platform(),packages=packages,
     devices=subprocess.check_output(['nvidia-smi','--query-gpu=index,name,driver_version,memory.total','--format=csv,noheader']).decode(),
     role='single shared existing environment for task inference and offline analysis; no CUDA context created by this inventory'))
