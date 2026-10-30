"""Freeze the original BDD fit-only RoIs for the optional native VK checkpoint."""
import json
from pathlib import Path
import numpy as np
from evidence_cpu import ROOT,DOMAINS,INPUTS,dump
import ics_probe_analyze as p

records=[]
for domain in DOMAINS:
    with np.load(str(INPUTS['HF']/(domain+'.npz')),allow_pickle=False) as f:
        mask=(f['category']==1)&p.split_mask(f['video'],0)
        expected={(int(i),int(t),tuple(map(float,b))) for i,t,b in zip(f['image_id'][mask],f['instance_id'][mask],f['bbox'][mask])}
        ids=set(f['image_id'][mask].tolist())
    source=Path('/data1/liuyeqiang/damot_rebuttal_20261026/data/annotations')/(domain+'.json')
    data=json.loads(source.read_text());car=next(c['id'] for c in data['categories'] if c['name'].lower()=='car')
    annotations=[]
    for a in data['annotations']:
        if a['category_id']!=car:continue
        x,y,w,h=a['bbox'];box=tuple(map(float,np.asarray([x,y,x+w,y+h],dtype=np.float32)))
        key=(int(a['image_id']),int(a['instance_id']),box)
        if key in expected:annotations.append(a)
    assert len(annotations)==len(expected),(domain,len(annotations),len(expected))
    images=[im for im in data['images'] if im['id'] in ids];vids=set(im['video_id'] for im in images)
    out=ROOT/'data/native_BDD_fit'/(domain+'.json')
    dump(out,dict(data,images=images,annotations=annotations,videos=[v for v in data['videos'] if v['id'] in vids]))
    records.append(dict(domain=domain,annotation=str(out),images=len(images),car_rois=len(annotations),videos=len(vids),
                        purpose='Exactly original HF car fit RoIs; native BDD features are used only to fit comparator bases'))
dump(ROOT/'manifests/native_BDD_fit_sampling.json',records)
print(json.dumps(records,indent=2))
