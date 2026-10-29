"""Verify all domains/stages use exactly the original F RoIs before analysis."""
import json,sys
from pathlib import Path
import numpy as np
from protocol import write_json
ROOT=Path(__file__).resolve().parents[1]
def check_pair(label,reference):
    rows=[]
    for domain in ['clear_holdout_45v','overcast','rainy','dawn_dusk','night']:
            with np.load(str(ROOT/'results/feats'/label/(domain+'.npz')),allow_pickle=False) as z,np.load(str(ROOT/'results/feats'/reference/(domain+'.npz')),allow_pickle=False) as f:
                fields=['video','instance_id','frame_id','image_id','category','bbox','weather']
                for key in fields:assert np.array_equal(z[key],f[key]),(label,domain,key)
                for key in ['feat_clean','feat_fap']:assert z[key].shape==f[key].shape and np.isfinite(z[key]).all(),(label,domain,key)
                rows.append(dict(label=label,domain=domain,rois=len(z['video']),same_metadata_fields=fields))
    return rows
def check(name):
    rows=[]
    for k in range(1,5):rows+=check_pair('%s_stage%d_e5'%(name,k),'F_stage%d_e5'%k)
    write_json(ROOT/'results/S1'/name/'ROI_alignment.json',dict(status='PASS',reference='F at the same stage',checks=rows))
    print('ROI ALIGNMENT PASS',name,len(rows),'domain/checkpoint files',flush=True)
if __name__=='__main__':check(sys.argv[1])
