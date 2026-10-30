"""Independent sanity checks for the reported CPU evidence and frozen inputs."""
import json
from pathlib import Path
import numpy as np
from evidence_cpu import ROOT, icc, sha, csvout
import ics_probe_analyze as p

checks=[]
for values,expected in [(np.array([1.,1.,3.,3.]),1.),(np.array([-1.,1.,-1.,1.]),0.)]:
    value=icc(values,np.array(['a','a','b','b']),np.array(['a','a','b','b']))['ICC']
    assert abs(value-expected)<1e-12
checks.append(dict(check='weighted ICC known shared/independent examples',status='PASS'))
for model in ['F','HF']:
    root=ROOT/'part1'/model
    assert json.loads((root/'complete.json').read_text())['status']=='complete'
    protocol=json.loads((root/'protocol.json').read_text())
    assert not set(protocol['fit_videos'])&set(protocol['test_videos'])
    for record in protocol['inputs']:
        assert sha(record['path'])==record['sha256']
    reference=Path('/data1/liuyeqiang/idapr_20261030_153059/A')/model/'bases'/('%s_stage4_nuisance_k2.npy'%model)
    angle=float(np.max(p.principal_angles_deg(np.load(str(root/'bases/INT_00.npy')),np.load(str(reference)))))
    assert angle<1e-4,(model,angle)
    checks.append(dict(check=model+' INT agrees with original idAPR stage4 fit basis; max angle %.8g deg'%angle,status='PASS'))
    for stem in ['B_shared_shifts','C_INT_crops','D_dose_response','D_dose_response_absolute','INT_quadrant']:
        for ext in ['png','pdf']:
            path=root/'figures'/(stem+'.'+ext)
            assert path.stat().st_size>1000
    checks.append(dict(check=model+' fit/test disjoint; input hashes unchanged; all plots exist',status='PASS'))
csvout(ROOT/'manifests/part1_validation.csv',checks)
print(json.dumps(checks,indent=2))
