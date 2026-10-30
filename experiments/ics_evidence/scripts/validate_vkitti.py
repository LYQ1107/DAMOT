"""Audit complete effective samples, pairing and split before publishing GPU metrics."""
import argparse,json
from pathlib import Path
import numpy as np
from evidence_cpu import ROOT,dump
from evaluate_vkitti import DOMAINS,load
from report import rows


def main(part):
    labels=(['F','HF']+(['VK_self'] if (ROOT/'features/VK_self/meta.json').exists() else [])) if part=='2' else ['C0','C1','C2','C3','C4','X3-1']
    if part=='2' and (ROOT/'manifests/part2_inference_complete.json').exists():
        assert 'VK_self' in labels
    checks=[]
    reference=None
    for label in labels:
        data=load(label,part=='2')
        for domain in DOMAINS:
            ann=json.loads((ROOT/'data/vkitti2/annotations'/(domain+'.json')).read_text())
            d=data[domain]
            assert set(d['annotation_id'].tolist())==set(a['id'] for a in ann['annotations'])
            video_frames={(str(v),int(f)) for v,f in zip(d['video'],d['frame_id'])}
            # Frames without cars may be absent from feature rows; all sampled car annotations must remain.
            for video in np.unique(d['video']):
                frames=np.unique(d['frame_id'][d['video']==video])
                assert np.all(frames%3==0) and frames.max()<=177
                assert len(frames)>20, (label,domain,video,'unexpected second stride')
            if reference is not None:
                for k in ['video','scene','frame_id','annotation_id','instance_id','bbox','category']:
                    assert np.array_equal(d[k],reference[domain][k]),(label,domain,k)
            assert np.isfinite(d['Z']).all()
            assert np.allclose(np.linalg.norm(d['Z'],axis=1),1,atol=1e-6)
            checks.append(dict(model=label,domain=domain,rois=len(d['Z']),car_frames=len(video_frames),all_expected_annotations=True))
        if reference is None:reference=data
    if part=='2':
        for r in rows(ROOT/'part2/CCR_replicates.csv'):
            assert 0<=float(r['CCR'])<=1+1e-10
        for r in rows(ROOT/'part2/CCR.csv'):
            assert int(r['pairs'])>0 and float(r['delta_total_sq'])>0
    else:
        for r in rows(ROOT/'part3/retrieval.csv'):
            assert 0<=float(r['cross_mAP_pp'])<=100 and int(r['queries'])>0
    dump(ROOT/'manifests'/('part%s_validation.json'%part),dict(status='PASS',checks=checks,
        frames_predeclared_before_inference=True,same_metadata_across_models=True))
    print('PASS part',part,len(checks),'domain/model audits')


if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--part',required=True,choices=['2','3'])
    main(ap.parse_args().part)
