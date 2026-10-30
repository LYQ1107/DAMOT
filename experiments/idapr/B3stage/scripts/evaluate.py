"""Full-domain native frozen evaluation, with cached per-video predictions."""
import argparse,pickle,time,copy,shutil
from functools import partial
from common import *
p=argparse.ArgumentParser();p.add_argument('--model',choices=['F','HF'],required=True);p.add_argument('--domain',required=True);p.add_argument('--beta',type=float,default=0);p.add_argument('--checkpoint');p.add_argument('--out');p.add_argument('--check-disabled',action='store_true');a=p.parse_args()
code=setup(a.model)
import torch,numpy as np
from mmcv import Config
from mmcv.cnn import fuse_conv_bn
from mmcv.parallel import collate,scatter
from torch.utils.data import DataLoader,Subset
from darth.models import build_model
from darth.datasets import build_dataset
from darth.datasets.eval.coco_eval_dataset import BDD100K
from darth.models.track_heads.ics_inference_track_head import ICSInferenceTrackHead
def all_frames(self):
    by={v['id']:[] for v in self.gt_data['videos']}
    for im in self.gt_data['images']:by[im['video_id']].append(im)
    return {v:{im['id']:i for i,im in enumerate(sorted(ims,key=lambda x:x['frame_id']))} for v,ims in by.items()}
BDD100K._compute_image_to_timestep_mappings=all_frames
torch.multiprocessing.set_start_method('fork',force=True);torch.backends.cudnn.benchmark=(a.model=='F');seed(20261029)
out=Path(a.out) if a.out else Q/'X2'/a.model/a.domain/('beta%d'%a.beta)
out.mkdir(parents=True,exist_ok=True)
if (out/'complete.json').exists():print('ALREADY_COMPLETE',out);sys.exit(0)
ckpath=Path(a.checkpoint) if a.checkpoint else checkpoint(a.model)
checksum=sha(ckpath)
cfgpath=(HF/'results/stage4'/a.domain/'used_config.py' if a.model=='HF' else RUN/'code/configs/mot/qdtrack/bdd_7domain'/('qdtrack_faster-rcnn_r50_fpn_20e_bdd_%s.py'%a.domain))
cfg=Config.fromfile(str(cfgpath))
cfg.cudnn_benchmark=(a.model=='F')
strip_init(cfg.model);cfg.model.tracker.init_score_thr=.30;cfg.model.tracker.obj_score_thr=.20;cfg.model.detector.test_cfg.rcnn.score_thr=.05
if 'test_cfg' in cfg.model.detector.roi_head:cfg.model.detector.roi_head.test_cfg.score_thr=.05
cfg.data.test.ann_file=str(RUN/'data/annotations'/(a.domain+'.json'))
local_views=Q/'data_views'/a.domain
cfg.data.test.img_prefix=str(local_views if local_views.is_dir() else RUN/'data/views'/a.domain)+'/'
cfg.data.test.classes=CLASSES;cfg.data.test.test_mode=True
original_cfg=copy.deepcopy(cfg.model)
cfg.model.track_head.type='ICSInferenceTrackHead'
cfg.model.track_head.inference_trim_beta=a.beta
cfg.model.track_head.inference_trim_basis=str(Q/'X2'/('basis_e_%s.npy'%a.model)) if (Q/'X2'/('basis_e_%s.npy'%a.model)).exists() else None
cfg.dump(str(out/'used_config.py'))
dump(out/'runtime.json',dict(cudnn_benchmark=torch.backends.cudnn.benchmark,workers=2,device=os.environ.get('CUDA_VISIBLE_DEVICES'),torch_version=torch.__version__,cudnn_version=torch.backends.cudnn.version(),python=sys.executable))
ds=build_dataset(cfg.data.test)
assert [ds.coco.cats[cid]['name'] for cid,label in sorted(ds.cat2label.items(),key=lambda x:x[1])]==list(CLASSES)
videos={}
for i,im in enumerate(ds.data_infos):videos.setdefault(im['video_id'],[]).append(i)
expected={'overcast':(22,4448),'rainy':(17,3150),'night':(36,7286),'dawn_dusk':(33,6476)}
assert (len(videos),len(ds))==expected[a.domain],('Evaluation pool changed',a.domain,len(videos),len(ds))
def load(modelcfg):
    ck=torch.load(str(ckpath),map_location='cpu');m=build_model(copy.deepcopy(modelcfg));target=m.state_dict()
    state={k:v for k,v in ck['state_dict'].items() if k in target}
    assert set(state)==set(target),(set(target)-set(state))
    m.load_state_dict(state,strict=True);m.CLASSES=CLASSES
    del state,ck
    m=fuse_conv_bn(m).cuda();m.eval();return m
model=load(cfg.model)
def equal(x,y):
    if isinstance(x,dict):return set(x)==set(y) and all(equal(x[k],y[k]) for k in x)
    if isinstance(x,(list,tuple)):return len(x)==len(y) and all(equal(u,v) for u,v in zip(x,y))
    return np.array_equal(x,y)
if a.check_disabled:
    assert a.beta==0
    baseline=load(original_cfg);ids=next(iter(videos.values()))[:16]
    loader=DataLoader(Subset(ds,ids),batch_size=1,shuffle=False,num_workers=2,collate_fn=partial(collate,samples_per_gpu=1))
    baseline.tracker.reset();model.tracker.reset()
    with torch.no_grad():
        for idx,b in zip(ids,loader):
            b=scatter(b,[0])[0];r1=baseline(return_loss=False,rescale=True,**b);r2=model(return_loss=False,rescale=True,**b)
            assert equal(r1,r2),('Disabled switch changed output',idx)
    dump(out/'disabled_bitwise_check.json',dict(frames=len(ids),outputs_bitwise_equal=True,comparison='native original class vs optional class beta=0, same checkpoint, fusion and inputs'))
    del baseline;torch.cuda.empty_cache()
def save_pickle(path,obj):
    path.parent.mkdir(parents=True,exist_ok=True);tmp=path.with_suffix('.tmp')
    with tmp.open('wb') as f:pickle.dump(obj,f,protocol=4)
    tmp.replace(path)
pending=[]
for v,ids in videos.items():
    path=out/'predictions'/('video_%s.pkl'%v)
    if path.exists():
        with path.open('rb') as f:r=pickle.load(f)
        assert r['indices']==ids and r['checkpoint_sha256']==checksum and r['beta']==a.beta
    else:pending.extend(ids)
loader=DataLoader(Subset(ds,pending),batch_size=1,shuffle=False,num_workers=2,collate_fn=partial(collate,samples_per_gpu=1))
active=None;outputs=[]
with torch.no_grad():
    for idx,b in zip(pending,loader):
        im=ds.data_infos[idx];v=im['video_id']
        if active!=v:
            assert im['frame_id']==0;model.tracker.reset();active=v;outputs=[]
        outputs.append(model(return_loss=False,rescale=True,**scatter(b,[0])[0]))
        if idx==videos[v][-1]:
            save_pickle(out/'predictions'/('video_%s.pkl'%v),dict(indices=videos[v],checkpoint_sha256=checksum,beta=a.beta,outputs=outputs))
            print('VIDEO_COMPLETE',a.model,a.domain,a.beta,v,len(outputs),flush=True)
del model;torch.cuda.empty_cache()
results=dict(det_bboxes=[],track_bboxes=[]);seen=[]
for v,ids in videos.items():
    with (out/'predictions'/('video_%s.pkl'%v)).open('rb') as f:r=pickle.load(f)
    assert r['indices']==ids;seen.extend(ids)
    for obj in r['outputs']:
        for k in results:results[k].append(obj[k])
assert seen==list(range(len(ds)))
save_pickle(out/'raw_results.pkl',results)
metrics=ds.evaluate(results,metric=['bbox','track'],results_path=str(out/'evaluation'),results_kwargs=dict(scalabel=False,append_empty_preds=True),track_kwargs=dict(iou_thr=.5,ignore_iof_thr=.5,ignore_by_classes=False,nproc=1,majority_voting=False))
dump(out/'evaluation_flat_metrics.json',metrics)
for kind in ['summary','hota','track']:shutil.copyfile(str(out/'evaluation'/(kind+'_metrics.csv')),str(out/(kind+'_metrics.csv')))
table={r['']:r for r in csv.DictReader((out/'summary_metrics.csv').open())}
rows=[]
reference=HF/'results/stage_domain_metrics.csv' if a.model=='HF' else RUN/'results/exp1/stage_domain_metrics.csv'
refs=[r for r in csv.DictReader(reference.open()) if int(r['stage'])==4 and r['eval_domain']==a.domain and (a.model=='HF' or r['model']=='F')]
checks=[]
for cat in list(CLASSES)+['OVERALL']:
    r=dict(model=a.model,domain=a.domain,beta=a.beta,category=cat,checkpoint=str(ckpath),checkpoint_sha256=checksum,frames=len(ds),videos=len(videos))
    for k in ['HOTA','DetA','IDF1','AssA']:r[k+'_percent']=float(table[cat][k])
    rows.append(r)
    if a.beta==0 and not a.checkpoint and any(x['category']==cat for x in refs):
        ref=next(x for x in refs if x['category']==cat)
        for k in ['HOTA','DetA','IDF1','AssA']:
            delta=r[k+'_percent']-float(ref[k+'_percent']);checks.append(dict(category=cat,metric=k,observed=r[k+'_percent'],reference=float(ref[k+'_percent']),delta=delta,passed=abs(delta)<1e-6))
write_csv(out/'metrics.csv',rows)
if checks:
    assert {'OVERALL','car'} <= {r['category'] for r in checks}
    write_csv(out/'baseline_parity.csv',checks)
    assert all(r['passed'] for r in checks),('Frozen baseline mismatch; intervention blocked',out)
dump(out/'complete.json',dict(status='complete',frames=len(ds),videos=len(videos),beta=a.beta,baseline_parity=all(r['passed'] for r in checks) if checks else None,reference=str(reference),all_frame_hota=True,time=time.time()))
print('EVAL_COMPLETE',out,flush=True)
