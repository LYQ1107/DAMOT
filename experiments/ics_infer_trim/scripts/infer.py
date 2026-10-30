"""Native inference; detector reuse across independent intervention trackers."""
import argparse,copy,pickle,time,fcntl
from functools import partial
from common import *

def equal(a,b):
    if isinstance(a,dict):return set(a)==set(b) and all(equal(a[k],b[k]) for k in a)
    if isinstance(a,(list,tuple)):return len(a)==len(b) and all(equal(x,y) for x,y in zip(a,b))
    x,y=np.asarray(a),np.asarray(b)
    return x.dtype==y.dtype and x.shape==y.shape and x.tobytes(order='C')==y.tobytes(order='C')

def save(path,value):
    path.parent.mkdir(parents=True,exist_ok=True);tmp=path.with_suffix('.tmp')
    with tmp.open('wb') as f:pickle.dump(value,f,protocol=4)
    tmp.replace(path)

def main():
    p=argparse.ArgumentParser();p.add_argument('--model',required=True);p.add_argument('--domain',required=True)
    p.add_argument('--preflight',action='store_true');p.add_argument('--checkpoint');p.add_argument('--source')
    p.add_argument('--cases-json');a=p.parse_args()
    assert nas_mounted(),'NAS is not mounted'
    lock=Path('/tmp/ics_trim_%s_%s.lock'%(a.model,a.domain)).open('w')
    fcntl.flock(lock,fcntl.LOCK_EX)
    if (ROOT/'jobs'/a.model/a.domain/'inference_complete.json').exists():
        print('ALREADY_COMPLETE',a.model,a.domain,flush=True);return
    import torch
    from mmcv import Config
    from mmcv.cnn import fuse_conv_bn
    from mmcv.parallel import collate,scatter
    from torch.utils.data import DataLoader,Subset
    from darth.models import build_model
    from darth.datasets import build_dataset
    from darth.datasets.eval.coco_eval_dataset import BDD100K
    from darth.models.track_heads.ics_inference_track_head import ICSInferenceTrackHead
    torch.set_num_threads(1);torch.cuda.set_per_process_memory_fraction(.35,0)
    torch.backends.cudnn.benchmark=False;torch.multiprocessing.set_start_method('fork',force=True);seed()
    def all_frames(self):
        by={v['id']:[] for v in self.gt_data['videos']}
        for im in self.gt_data['images']:by[im['video_id']].append(im)
        return {v:{im['id']:i for i,im in enumerate(sorted(ims,key=lambda x:x['frame_id']))} for v,ims in by.items()}
    BDD100K._compute_image_to_timestep_mappings=all_frames
    source=Path(a.source) if a.source else CROOT/'evaluation'/a.model/a.domain
    job=ROOT/'jobs'/a.model/a.domain;job.mkdir(parents=True,exist_ok=True)
    casespec=json.loads(Path(a.cases_json).read_text()) if a.cases_json else cases(a.model)
    assert casespec[0]['beta']==0
    if not a.preflight and (job/'inference_complete.json').exists():print('ALREADY_COMPLETE',job);return
    cfg=Config.fromfile(str(source/'used_config.py'));strip_init(cfg.model)
    cfg.model.track_head.type='ICSInferenceTrackHead'
    cfg.model.track_head.inference_trim_enabled=False;cfg.model.track_head.inference_trim_beta=0.
    cfg.model.track_head.inference_trim_basis=None
    cfg.model.tracker.init_score_thr=.30;cfg.model.tracker.obj_score_thr=.20
    cfg.model.detector.test_cfg.rcnn.score_thr=.05
    if 'test_cfg' in cfg.model.detector.roi_head:cfg.model.detector.roi_head.test_cfg.score_thr=.05
    cfg.cudnn_benchmark=False;cfg.data.test.test_mode=True
    ds=build_dataset(cfg.data.test)
    videos={}
    for i,im in enumerate(ds.data_infos):videos.setdefault(im['video_id'],[]).append(i)
    ckpath=Path(a.checkpoint) if a.checkpoint else checkpoint(a.model);checksum=sha(ckpath)
    ck=torch.load(str(ckpath),map_location='cpu');model=build_model(copy.deepcopy(cfg.model));target=model.state_dict()
    state={k:v for k,v in ck['state_dict'].items() if k in target}
    assert set(state)==set(target),sorted(set(target)-set(state));model.load_state_dict(state,strict=True)
    alpha=None
    text=ck.get('meta',{}).get('config')
    if isinstance(text,str):
        try:alpha=Config.fromstring(text,'.py').model.track_head.get('causal_trim_alpha',0.)
        except Exception as exc:raise RuntimeError('Cannot recover checkpoint model alpha: '+str(exc))
    model.track_head.load_checkpoint_bank(ck['state_dict'].get('track_head.causal_bank.pcs'),alpha)
    if alpha is not None:cfg.model.track_head.causal_trim_alpha=float(alpha)
    ignored=sorted(set(ck['state_dict'])-set(target));del state,ck,target
    model.CLASSES=CLASSES;model=fuse_conv_bn(model).cuda();model.eval()
    trackers={c['label']:copy.deepcopy(model.tracker) for c in casespec}
    # Detector features/proposals/results are invariant to association shrinkage.
    # Every case still runs the native model.forward and has an independent tracker.
    cache={};calls={}
    def memo(obj,name):
        native=getattr(obj,name);calls[name]=0
        def wrapped(*args,**kwargs):
            if name not in cache:
                cache[name]=native(*args,**kwargs);calls[name]+=1
            return copy.deepcopy(cache[name]) if name=='simple_test' else cache[name]
        setattr(obj,name,wrapped)
    memo(model.detector,'extract_feat');memo(model.detector.rpn_head,'simple_test_rpn');memo(model.detector.roi_head,'simple_test')
    dump(job/'provenance.json',dict(checkpoint=str(ckpath),checkpoint_sha256=checksum,source=str(source),
         config_alpha=alpha,cases=casespec,ignored_checkpoint_keys=ignored,
         raw_embedding_operator='z - beta * (z @ E) @ E.T; QR columns; no normalization or min_samples gate',
         device=os.environ.get('CUDA_VISIBLE_DEVICES'),torch=torch.__version__,cudnn=torch.backends.cudnn.version(),
         cudnn_benchmark=False,seed=20261029,workers=2,memory_fraction=.35))
    for case in casespec:
        out=ROOT/'results'/case['label']/a.domain;out.mkdir(parents=True,exist_ok=True)
        casecfg=copy.deepcopy(cfg)
        casecfg.model.track_head.update(inference_trim_enabled=case['beta']!=0.,inference_trim_beta=case['beta'],
                                        basis_source=case['basis_source'],inference_trim_basis=case.get('npy_path'))
        casecfg.dump(str(out/'used_config.py'))
    pending=[]
    for vid,ids in videos.items():
        complete=True
        for case in casespec:
            file=ROOT/'results'/case['label']/a.domain/'predictions'/('video_%s.pkl'%vid)
            if not file.exists():complete=False;break
            with file.open('rb') as f:r=pickle.load(f)
            assert r['indices']==ids and r['checkpoint_sha256']==checksum and r['beta']==case['beta']
        if not complete:pending.extend(ids)
    if a.preflight:pending=next(iter(videos.values()))[:16]
    loader=DataLoader(Subset(ds,pending),batch_size=1,shuffle=False,num_workers=2,collate_fn=partial(collate,samples_per_gpu=1))
    active=None;original=None;outputs={};checked=0;start=time.time()
    with torch.no_grad():
        for idx,batch in zip(pending,loader):
            assert nas_mounted(),'NAS disconnected'
            im=ds.data_infos[idx];vid=im['video_id']
            if active!=vid:
                assert im['frame_id']==0;active=vid
                for tracker in trackers.values():tracker.reset()
                outputs={c['label']:[] for c in casespec}
                with (source/'predictions'/('video_%s.pkl'%vid)).open('rb') as f:original=pickle.load(f)
                assert original['indices']==videos[vid] and original['checkpoint_sha256']==checksum
            cache.clear();data=scatter(batch,[0])[0]
            for ci,case in enumerate(casespec[:1] if a.preflight else casespec):
                model.tracker=trackers[case['label']]
                model.track_head.set_inference_intervention(beta=case['beta'],basis_source=case['basis_source'],npy_path=case.get('npy_path'),enabled=case['beta']!=0)
                prediction=model(return_loss=False,rescale=True,**data)
                if ci==0:
                    pos=videos[vid].index(idx)
                    if not equal(prediction,original['outputs'][pos]):
                        dump(job/'parity_failure.json',dict(video=vid,index=idx,frame_id=im['frame_id'],status='blocked_before_accepting_interventions'))
                        raise AssertionError(('Beta0 differs from frozen predictions',a.model,a.domain,vid,idx))
                    checked+=1
                outputs[case['label']].append(prediction)
            cache.clear()
            if not a.preflight and idx==videos[vid][-1]:
                for case in casespec:
                    save(ROOT/'results'/case['label']/a.domain/'predictions'/('video_%s.pkl'%vid),
                         dict(indices=videos[vid],checkpoint_sha256=checksum,beta=case['beta'],outputs=outputs[case['label']]))
                print('VIDEO_COMPLETE',a.model,a.domain,vid,len(videos[vid]),'elapsed',round(time.time()-start,1),flush=True)
    assert all(n==len(pending) for n in calls.values()),calls
    if a.preflight:
        dump(job/'preflight.json',dict(status='PASS',frames=checked,det_and_track_bitwise_equal_to_frozen=True,all_frame_proof_pending=True));print('PREFLIGHT_PASS',job,checked,flush=True);return
    for case in casespec:
        results=dict(det_bboxes=[],track_bboxes=[]);seen=[]
        for vid,ids in videos.items():
            with (ROOT/'results'/case['label']/a.domain/'predictions'/('video_%s.pkl'%vid)).open('rb') as f:r=pickle.load(f)
            assert r['indices']==ids;seen.extend(ids)
            for pred in r['outputs']:
                for k in results:results[k].append(pred[k])
        assert seen==list(range(len(ds)));save(ROOT/'results'/case['label']/a.domain/'raw_results.pkl',results)
    # Also verify resumed baseline videos, so completion always certifies the entire domain.
    for vid,ids in videos.items():
        with (source/'predictions'/('video_%s.pkl'%vid)).open('rb') as f:old=pickle.load(f)
        with (ROOT/'results'/casespec[0]['label']/a.domain/'predictions'/('video_%s.pkl'%vid)).open('rb') as f:new=pickle.load(f)
        assert equal(old['outputs'],new['outputs']),('Resumed parity failure',vid)
    dump(job/'inference_complete.json',dict(status='complete',frames=len(ds),videos=len(videos),
         beta0_bitwise_equal_to_frozen_all_frames=True,detector_calls_this_invocation=calls,
         trimmed_embedding_calls=model.track_head.inference_trim_calls,min_trimmed_roi_count=model.track_head.inference_trim_min_rois,
         peak_cuda_allocated_bytes=torch.cuda.max_memory_allocated(),peak_cuda_reserved_bytes=torch.cuda.max_memory_reserved(),
         elapsed_seconds=time.time()-start,checkpoint_sha256=checksum))
    print('INFERENCE_COMPLETE',job,flush=True)

if __name__=='__main__':main()
