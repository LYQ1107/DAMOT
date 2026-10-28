import os,sys,json,time,hashlib,copy,pickle
from pathlib import Path
RUN=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(RUN/'code'))
# Import torch first: the inherited MKL and torch runtime otherwise conflict.
import torch
torch.set_num_threads(1)
import numpy as np,cv2,mmcv
cv2.setNumThreads(0)
from mmcv import Config
from mmcv.cnn import fuse_conv_bn
import darth.datasets,darth.models
from darth.datasets import build_dataset
from darth.models import build_model

CLASSES=('pedestrian','car','truck','bus','motorcycle','bicycle')
def dump(p,obj):
    p=Path(p);p.parent.mkdir(parents=True,exist_ok=True);q=p.with_suffix(p.suffix+'.tmp');q.write_text(json.dumps(obj,indent=2,default=str,ensure_ascii=False));q.replace(p)
def save_pickle(p,obj):
    p=Path(p);p.parent.mkdir(parents=True,exist_ok=True);q=p.with_suffix(p.suffix+'.tmp')
    with open(q,'wb') as f:pickle.dump(obj,f,protocol=4)
    q.replace(p)
def fixed_seed(seed=20261026):
    import random
    random.seed(seed);np.random.seed(seed);torch.manual_seed(seed);torch.cuda.manual_seed_all(seed)
def strip_init(obj):
    if isinstance(obj,dict):
        obj.pop('init_cfg',None);obj.pop('pretrained',None)
        for v in obj.values():strip_init(v)
    elif isinstance(obj,(list,tuple)):
        for v in obj:strip_init(v)
def eval_config(domain):
    c=Config.fromfile(str(RUN/'code/configs/mot/qdtrack/bdd_7domain/qdtrack_faster-rcnn_r50_fpn_20e_bdd_overcast.py'))
    c.model.tracker.init_score_thr=.30;c.model.tracker.obj_score_thr=.20
    c.model.detector.test_cfg.rcnn.score_thr=.05
    if 'test_cfg' in c.model.detector.roi_head:c.model.detector.roi_head.test_cfg.score_thr=.05
    strip_init(c.model)
    c.data.test.ann_file=str(RUN/'data/annotations'/(domain+'.json'))
    c.data.test.img_prefix=str(RUN/'cache/images'/domain)+'/'
    c.data.test.test_mode=True;c.data.test.classes=CLASSES
    return c
def dataset(domain):
    c=eval_config(domain);ds=build_dataset(c.data.test)
    assert list(ds.cat2label.values())==list(range(6)),ds.cat2label
    assert [ds.coco.cats[cid]['name'] for cid,label in sorted(ds.cat2label.items(),key=lambda v:v[1])]==list(CLASSES)
    return ds
def load_model(label,deploy=True,fusion=False,device='cuda'):
    c=eval_config('clear_holdout_45v') if deploy else Config.fromfile(str(RUN/'configs'/(label+'_embedded.py')))
    strip_init(c.model)
    model=build_model(copy.deepcopy(c.model))
    path=RUN/'cache/weights'/(label+'.pth');ck=torch.load(str(path),map_location='cpu')
    state=ck['state_dict'];target=model.state_dict()
    chosen={k:v for k,v in state.items() if k in target}
    missing=set(target)-set(chosen)
    # torch versions may synthesize num_batches_tracked, but all actual checkpoint
    # keys are required here, including backbone, detector, association and teacher.
    assert not missing,(label,sorted(missing))
    for k in target:assert target[k].shape==chosen[k].shape,(k,target[k].shape,chosen[k].shape)
    model.load_state_dict(chosen,strict=True)
    for name in ['detector.backbone.conv1.weight','detector.roi_head.bbox_head.fc_cls.weight','track_head.embed_head.fc_embed.weight']:
        if name in chosen:assert torch.equal(model.state_dict()[name],state[name])
    if not deploy:
        assert any(k.startswith('teacher_detector.') for k in chosen)
        assert torch.equal(model.teacher_detector.backbone.conv1.weight,state['teacher_detector.backbone.conv1.weight'])
        assert not model.roi_proto_memory_summary()['history_domains']
    record=dict(model=label,checkpoint=str(path),deployment=deploy,matched_keys=len(chosen),target_keys=len(target),ignored_keys=[k for k in state if k not in chosen],missing_keys=[],fusion=fusion,teacher_loaded=not deploy,CLASSES=ck['meta']['CLASSES'])
    model.CLASSES=CLASSES
    del ck,state,chosen
    if fusion:model=fuse_conv_bn(model)
    model=model.to(device);model.eval()
    dump(RUN/'manifests'/('load_%s_%s.json'%(label,'deploy' if deploy else 'adapter')),record)
    c.dump(str(RUN/'configs'/('%s_%s_resolved.py'%(label,'deploy' if deploy else 'adapter'))))
    return model,c

def preprocessed_frame(path,info,training_base=False,flip=False):
    im=mmcv.imread(str(path)).astype(np.float32)
    assert im is not None,path
    # Both the archived base training and native evaluation resize to 1296x720.
    im,scale=mmcv.imrescale(im,(1296,720),return_scale=True)
    shape=im.shape
    if flip:im=mmcv.imflip(im)
    meta=dict(filename=str(path),ori_filename=info['file_name'],ori_shape=(info['height'],info['width'],3),img_shape=shape,frame_id=info['frame_id'],video_id=info['video_id'],scale_factor=np.array([shape[1]/info['width'],shape[0]/info['height']]*2,dtype=np.float32),flip=flip,flip_direction='horizontal',img_norm_cfg=dict(mean=np.array([123.675,116.28,103.53]),std=np.array([58.395,57.12,57.375]),to_rgb=True))
    if training_base:return torch.from_numpy(im.transpose(2,0,1).copy()).unsqueeze(0),[meta]
    im=mmcv.imnormalize(im,np.array([123.675,116.28,103.53]),np.array([58.395,57.12,57.375]),to_rgb=True)
    im=mmcv.impad_to_multiple(im,32);meta['pad_shape']=im.shape
    return torch.from_numpy(im.transpose(2,0,1).copy()).unsqueeze(0),[meta]

def write_csv(path,rows):
    import csv
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    with open(path,'w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
