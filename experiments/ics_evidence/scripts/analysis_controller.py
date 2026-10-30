"""Run offline evaluation as each frozen inference stage becomes available."""
import os
import subprocess
import time
from evidence_cpu import ROOT,dump

PY='/data1/liuyeqiang/damot_rebuttal_20261026/env/bin/python'


def main():
    for part in ['2','3']:
        marker=ROOT/'manifests'/('part%s_inference_complete.json'%part)
        while not marker.exists():time.sleep(15)
        with (ROOT/'logs'/('part%s_analysis.log'%part)).open('w') as f:
            subprocess.run([PY,'-u',str(ROOT/'scripts/evaluate_vkitti.py'),'--part',part],stdout=f,stderr=subprocess.STDOUT,check=True)
            subprocess.run([PY,'-u',str(ROOT/'scripts/validate_vkitti.py'),'--part',part],stdout=f,stderr=subprocess.STDOUT,check=True)
        subprocess.run([PY,str(ROOT/'scripts/report.py')],check=True)
        subprocess.run([PY,str(ROOT/'scripts/build_delivery.py')],check=True)
        dump(ROOT/'manifests'/('part%s_ready_to_publish.json'%part),dict(status='ready',time=time.time()))
        print('ANALYSIS_COMPLETE',part,flush=True)


if __name__=='__main__':main()
