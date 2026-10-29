"""Check nonzero cached replay against the actual enabled head and detector."""
import argparse,json,sys
from pathlib import Path
from functools import partial
from types import SimpleNamespace
import ics_tracking_intervention as s4
from apr_complement_experiment import torch,np,build_model,build_dataset
from darth.models.track_heads.inference_subspace_track_head import project_association_features
from mmtrack.models import build_tracker
from mmtrack.core import outs2results
from mmcv.cnn import fuse_conv_bn
from mmcv.parallel import collate,scatter
from torch.utils.data import DataLoader,Subset
ROOT=Path(__file__).resolve().parents[3];sys.path.insert(0,str(ROOT/'scripts'))
from protocol import write_json

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--model',choices=['F','HF'],required=True);a=ap.parse_args()
    assert all((ROOT/'results/S4'/a.model/'baseline'/d/'complete.json').exists() for d in ['overcast','rainy','dawn_dusk','night'])
    torch.backends.cudnn.benchmark=False;reports=[]
    for basis_name,beta in [('ICS_stored_m2',.5),('Identity_fit_m16',1.)]:
        basis_file=ROOT/'results'/('analysis_%s_stage4_e5_v2'%a.model)/'bases'/(basis_name+'.npy')
        cfg=s4.config('overcast');cfg.model.track_head.subspace_intervention=dict(enabled=True,beta=beta,basis_file=str(basis_file));ds=build_dataset(cfg.data.test)
        video=ds.data_infos[0]['video_id'];indices=[i for i,im in enumerate(ds.data_infos) if im['video_id']==video]
        records=s4.read(ROOT/'cache/S4'/a.model/'overcast'/('video_%s.pkl.gz'%video));assert [r['index'] for r in records]==indices
        model=build_model(cfg.model);ck=torch.load(str(s4.checkpoint(a.model)),map_location='cpu');target=model.state_dict();state={k:v for k,v in ck['state_dict'].items() if k in target};assert set(state)==set(target);model.load_state_dict(state,strict=True);del ck,state,target
        model.CLASSES=s4.CLASSES;model=fuse_conv_bn(model).cuda();model.eval();model.tracker.reset();tracker=build_tracker(cfg.model.tracker)
        E=torch.from_numpy(np.load(str(basis_file)).astype(np.float32)).cuda();current={};max_error=0.;exact_features=True
        orig_head=model.track_head.extract_bbox_feats
        def inspected(x,boxes):
            value=orig_head(x,boxes);reference=project_association_features(torch.from_numpy(current['record']['embedding']).cuda(),E,beta)
            current['max_abs_error']=float((value-reference).abs().max().item()) if value.numel() else 0.;current['exact']=torch.equal(value,reference)
            return value
        model.track_head.extract_bbox_feats=inspected
        loader=DataLoader(Subset(ds,indices),batch_size=1,shuffle=False,num_workers=2,collate_fn=partial(collate,samples_per_gpu=1))
        with torch.no_grad():
            for rec,batch in zip(records,loader):
                current.clear();current['record']=rec
                native=model(return_loss=False,rescale=True,**scatter(batch,[0])[0])
                max_error=max(max_error,current.get('max_abs_error',0.));exact_features &=current.get('exact',True)
                def pooled(x,boxes):
                    assert np.array_equal(boxes[0].cpu().numpy(),rec['pool_boxes'])
                    return project_association_features(torch.from_numpy(rec['embedding']).cuda(),E,beta)
                fake=SimpleNamespace(track_head=SimpleNamespace(extract_bbox_feats=pooled))
                boxes,labels,ids=tracker.track(img_metas=rec['img_metas'],feats=None,model=fake,bboxes=torch.from_numpy(rec['bboxes']).cuda(),labels=torch.from_numpy(rec['labels']).cuda(),frame_id=rec['frame_id'])
                predicted=outs2results(bboxes=boxes,labels=labels,ids=ids,num_classes=6)['bbox_results']
                assert all(np.array_equal(x,y) for x,y in zip(predicted,native['track_bboxes'])),(a.model,basis_name,rec['frame_id'])
                assert all(np.array_equal(x,y) for x,y in zip(native['det_bboxes'],rec['output']['det_bboxes']))
        reports.append(dict(basis=basis_name,beta=beta,video=video,frames=len(records),all_six_class_track_outputs_exact=True,all_detection_outputs_exact=True,projected_features_bit_exact=bool(exact_features),max_projected_feature_abs_error=max_error))
        print('NATIVE PROJECTION CHECK',a.model,reports[-1],flush=True)
        del model,tracker;torch.cuda.empty_cache()
    write_json(ROOT/'results/S4'/a.model/'native_projection_validation.json',dict(status='PASS',selection='first overcast video, predeclared bases ICS2 beta.5 and identity16 beta1',tests=reports))
if __name__=='__main__':main()
