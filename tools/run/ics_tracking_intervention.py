"""S4 native FP32 capture and exact native-tracker replay (one GPU).

Only detections and pre-tracker embeddings are cached. Beta-zero replay must
match every native track output bit-for-bit and all archived metrics before
nonzero interventions are allowed. No post-projection renormalization.
"""
import argparse,copy,csv,gzip,json,os,pickle,sys
from pathlib import Path
from functools import partial
from types import SimpleNamespace
from apr_complement_experiment import Config,torch,np,strip_init,build_model,build_dataset,sha,CLASSES,EXPECTED,STAGES,dump
from evaluate_apr_complement import all_frame_hota_index,csv_write
from darth.models.track_heads.inference_subspace_track_head import project_association_features
from mmtrack.core import outs2results
ROOT=Path(__file__).resolve().parents[3]
OLD=Path('/data1/liuyeqiang/damot_rebuttal_20261026')
HF=Path('/data1/liuyeqiang/damot_apr_complement_hf_20261028_1420')

def save(path,obj):
    path.parent.mkdir(parents=True,exist_ok=True)
    temp=path.with_suffix(path.suffix+'.tmp')
    with gzip.open(str(temp),'wb',compresslevel=1) as f:pickle.dump(obj,f,protocol=4)
    temp.replace(path)

def read(path):
    with gzip.open(str(path),'rb') as f:return pickle.load(f)

def config(domain):
    cfg=Config.fromfile(str(ROOT/'code/configs/mot/qdtrack/bdd_7domain'/('qdtrack_faster-rcnn_r50_fpn_20e_bdd_%s.py'%domain)))
    strip_init(cfg.model);cfg.model.tracker.init_score_thr=.30;cfg.model.tracker.obj_score_thr=.20
    cfg.model.detector.test_cfg.rcnn.score_thr=.05
    if 'test_cfg' in cfg.model.detector.roi_head:cfg.model.detector.roi_head.test_cfg.score_thr=.05
    cfg.model.track_head.type='SubspaceInterventionQuasiDenseTrackHead'
    cfg.model.track_head.subspace_intervention=dict(enabled=False,beta=0.)
    data=json.loads((HF/'manifests/data.json').read_text())[domain]
    cfg.data.test.ann_file=data['annotation'];cfg.data.test.img_prefix=data['image_view']+'/'
    cfg.data.test.classes=CLASSES;cfg.data.test.test_mode=True
    return cfg

def checkpoint(name):
    return OLD/'cache/weights/F_stage4_e5.pth' if name=='F' else HF/'episode/adapt/stage04_after_night/epoch_5.pth'

def capture(name,domain):
    from torch.utils.data import DataLoader,Subset
    from mmcv.parallel import collate,scatter
    from mmcv.cnn import fuse_conv_bn
    folder=ROOT/'cache/S4'/name/domain;folder.mkdir(parents=True,exist_ok=True)
    if (folder/'complete.json').exists():return
    cfg=config(domain);cfg.dump(str(folder/'resolved_config.py'));ds=build_dataset(cfg.data.test)
    vids={}
    for i,im in enumerate(ds.data_infos):vids.setdefault(im['video_id'],[]).append(i)
    assert (len(vids),len(ds))==EXPECTED[domain]
    pending=[i for v,indices in vids.items() if not (folder/('video_%s.pkl.gz'%v)).exists() for i in indices]
    if pending:
        ck=torch.load(str(checkpoint(name)),map_location='cpu');model=build_model(cfg.model);target=model.state_dict();state={k:v for k,v in ck['state_dict'].items() if k in target};assert set(state)==set(target);model.load_state_dict(state,strict=True)
        dump(folder/'load.json',dict(checkpoint=str(checkpoint(name)),sha256=sha(checkpoint(name)),loaded=len(state),target=len(target),missing=[],fusion=True))
        del ck,state;model.CLASSES=CLASSES;model=fuse_conv_bn(model).cuda();model.eval()
        head=model.track_head;orig_head=head.extract_bbox_feats;orig_track=model.tracker.track;current={}
        def emb(x,boxes):
            value=orig_head(x,boxes);current['embedding']=value.detach().cpu().numpy();current['pool_boxes']=boxes[0].detach().cpu().numpy();return value
        def track(**kw):
            current.update(bboxes=kw['bboxes'].detach().cpu().numpy(),labels=kw['labels'].detach().cpu().numpy(),img_metas=copy.deepcopy(kw['img_metas']),frame_id=kw['frame_id'])
            return orig_track(**kw)
        head.extract_bbox_feats=emb;model.tracker.track=track
        loader=DataLoader(Subset(ds,pending),batch_size=1,shuffle=False,num_workers=2,collate_fn=partial(collate,samples_per_gpu=1))
        active=None;records=[]
        with torch.no_grad():
            for index,batch in zip(pending,loader):
                info=ds.data_infos[index];v=info['video_id']
                if active!=v:assert info['frame_id']==0;model.tracker.reset();active=v;records=[]
                current.clear();output=model(return_loss=False,rescale=True,**scatter(batch,[0])[0]);record=dict(current,index=index,output=output)
                if 'embedding' not in record:record.update(embedding=np.zeros((0,256),np.float32),pool_boxes=np.zeros((0,4),np.float32))
                records.append(record)
                if index==vids[v][-1]:save(folder/('video_%s.pkl.gz'%v),records);print('CAPTURE',name,domain,v,len(records),flush=True)
        del model;torch.cuda.empty_cache()
    dump(folder/'complete.json',dict(checkpoint=str(checkpoint(name)),videos=len(vids),frames=len(ds),precision='FP32',source='native simple_test; detections and embeddings cached without quantization'))

def archived_expected(name,domain):
    path=OLD/'results/exp1/stage_domain_metrics.csv' if name=='F' else HF/('results/stage4/%s/metrics.csv'%domain)
    rows=list(csv.DictReader(path.open()))
    return {r['category']:r for r in rows if (name!='F' or r['model']=='F') and int(r.get('stage',4))==4 and r['eval_domain']==domain}

def expected(name,domain):
    if name=='F' and (ROOT/'manifests/S4_F_fresh_baseline_authorization.json').exists():
        folder=ROOT/'results/S4/F/baseline_audit'/domain
        assert (folder/'complete.json').exists(),folder
        return {r['category']:r for r in csv.DictReader((folder/'metrics.csv').open())}
    return archived_expected(name,domain)

def replay(name,domain,basis_name,beta):
    from darth.datasets.eval.coco_eval_dataset import BDD100K
    BDD100K._compute_image_to_timestep_mappings=all_frame_hota_index
    tag='baseline' if beta==0 else '%s_beta%g'%(basis_name,beta)
    out=ROOT/'results/S4'/name/tag/domain;out.mkdir(parents=True,exist_ok=True)
    if (out/'complete.json').exists():return
    if beta:assert (ROOT/'results/S4'/name/'baseline'/domain/'complete.json').exists()
    cfg=config(domain);ds=build_dataset(cfg.data.test);tracker=__import__('mmtrack.models',fromlist=['build_tracker']).build_tracker(cfg.model.tracker)
    basis_file=ROOT/'results'/('analysis_%s_stage4_e5_v2'%name)/'bases'/(basis_name+'.npy')
    basis=torch.from_numpy(np.load(str(basis_file)).astype(np.float32)).cuda() if beta else None
    cfg.model.track_head.subspace_intervention=dict(enabled=bool(beta),beta=beta,basis_file=str(basis_file) if beta else None)
    results=dict(det_bboxes=[],track_bboxes=[]);frames=0;vids={}
    for i,im in enumerate(ds.data_infos):vids.setdefault(im['video_id'],[]).append(i)
    with torch.no_grad():
        for v,indices in vids.items():
            tracker.reset();cached=read(ROOT/'cache/S4'/name/domain/('video_%s.pkl.gz'%v));assert [r['index'] for r in cached]==indices
            for rec in cached:
                def pooled(x,boxes):
                    assert np.array_equal(boxes[0].cpu().numpy(),rec['pool_boxes'])
                    return project_association_features(torch.from_numpy(rec['embedding']).cuda(),basis,beta)
                model=SimpleNamespace(track_head=SimpleNamespace(extract_bbox_feats=pooled))
                boxes,labels,ids=tracker.track(img_metas=rec['img_metas'],feats=None,model=model,bboxes=torch.from_numpy(rec['bboxes']).cuda(),labels=torch.from_numpy(rec['labels']).cuda(),frame_id=rec['frame_id'])
                tracks=outs2results(bboxes=boxes,labels=labels,ids=ids,num_classes=6)['bbox_results']
                if beta==0:
                    assert all(np.array_equal(a,b) for a,b in zip(tracks,rec['output']['track_bboxes'])),(name,domain,v,rec['frame_id'])
                results['det_bboxes'].append(rec['output']['det_bboxes']);results['track_bboxes'].append(tracks);frames+=1
    assert frames==len(ds)
    save(out/'predictions.pkl.gz',results)
    dump(out/'config_diff.json',dict(checkpoint=str(checkpoint(name)),head_type=cfg.model.track_head.type,beta=beta,basis=basis_name,basis_sha256=sha(basis_file) if beta else None,renormalization=False,tracker=dict(cfg.model.tracker),all_other_parameters_unchanged=True,cache_replay='Native tracker receives exact FP32 embeddings and detections; beta0 per-frame outputs asserted identical'))
    cfg.dump(str(out/'resolved_config.py'))
    metrics=ds.evaluate(results,metric=['track'],results_path=str(out/'evaluation'),results_kwargs=dict(scalabel=False,append_empty_preds=True),track_kwargs=dict(iou_thr=.5,ignore_iof_thr=.5,ignore_by_classes=False,nproc=1,majority_voting=False))
    summary=list(csv.DictReader((out/'evaluation/summary_metrics.csv').open()));baseline=expected(name,domain) if beta==0 else {r['category']:r for r in csv.DictReader((ROOT/'results/S4'/name/'baseline'/domain/'metrics.csv').open())}
    rows=[]
    for r in summary:
        category=r[''];
        if category not in ['car','OVERALL']:continue
        row=dict(checkpoint=name,domain=domain,basis=basis_name,beta=beta,category=category,frames=frames)
        for m in ['HOTA','DetA','IDF1','AssA']:
            val=float(r[m]);base=float(baseline[category][m+'_percent']);row[m+'_percent']=val;row['delta_'+m+'_pp']=val-base
            if beta==0:assert abs(val-base)<1e-6,(name,domain,category,m,val,base)
        rows.append(row)
    csv_write(out/'metrics.csv',rows);dump(out/'complete.json',dict(status='complete',frames=frames,beta0_native_outputs_exact=beta==0,beta0_reference_metrics_exact=beta==0,reference='user-authorized fresh F baseline; historical differences preserved in baseline_audit' if name=='F' and (ROOT/'manifests/S4_F_fresh_baseline_authorization.json').exists() else 'frozen archived metrics',reference_categories_checked=['car','OVERALL']))
    print('S4 COMPLETE',name,tag,domain,flush=True)

def main():
    a=argparse.ArgumentParser();a.add_argument('--model',choices=['F','HF'],required=True);a.add_argument('--capture-only',action='store_true');a.add_argument('--baseline-only',action='store_true');args=a.parse_args();torch.backends.cudnn.benchmark=False
    for d in STAGES:capture(args.model,d)
    if args.capture_only:return
    for d in STAGES:replay(args.model,d,'baseline',0.)
    if args.baseline_only:return
    for b in ['ICS_stored_m2','FAP_clear_fit_m2','Identity_fit_m16','Random_m2']:
        for beta in [.5,1.]:
            for d in STAGES:replay(args.model,d,b,beta)

if __name__=='__main__':main()
