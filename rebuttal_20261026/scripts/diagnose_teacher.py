#!/usr/bin/env python3
import argparse,os,time,pickle
from collections import defaultdict
from runtime import *

class NightFrames(torch.utils.data.Dataset):
    def __init__(self,infos):self.infos=infos
    def __len__(self):return len(self.infos)
    def __getitem__(self,i):
        im=self.infos[i];x,meta=preprocessed_frame(RUN/'cache/images/night'/im['file_name'],im)
        return x,meta
def identity(batch):return batch[0]

def load_teacher(label):
    from mmdet.models import build_detector
    c=Config.fromfile(str(RUN/'configs'/(label+'_embedded.py')));strip_init(c.model)
    detector=build_detector(copy.deepcopy(c.model.detector))
    ck=torch.load(str(RUN/'cache/weights'/(label+'.pth')),map_location='cpu');state={k[len('teacher_detector.'):]:v for k,v in ck['state_dict'].items() if k.startswith('teacher_detector.')}
    assert len(state)==len(detector.state_dict()) and state
    detector.load_state_dict(state,strict=True)
    assert torch.equal(detector.backbone.conv1.weight,state['backbone.conv1.weight'])
    dump(RUN/'manifests'/('load_%s_teacher.json'%label),dict(checkpoint=label,branch='teacher_detector',strict=True,matched_keys=len(state),conf_thr=c.model.conf_thr,test_cfg=c.model.detector.test_cfg,roi_test_cfg=c.model.detector.roi_head.test_cfg))
    detector.cuda().eval()
    for v in detector.parameters():v.requires_grad_(False)
    return detector,c

def overlaps(b,g,iof=False):
    if not len(b) or not len(g):return np.zeros((len(b),len(g)))
    lt=np.maximum(b[:,None,:2],g[None,:,:2]);rb=np.minimum(b[:,None,2:4],g[None,:,2:4]);inter=np.prod(np.maximum(rb-lt,0),axis=2)
    ba=np.prod(np.maximum(b[:,2:4]-b[:,:2],0),axis=1)
    ga=np.prod(np.maximum(g[:,2:4]-g[:,:2],0),axis=1)
    den=ba[:,None] if iof else ba[:,None]+ga[None,:]-inter
    return inter/np.maximum(den,1e-12)

def match_image(pred,ann):
    rows=[];details=[]
    for cid in range(6):
        g=ann['bboxes'][ann['labels']==cid];p=pred[cid]
        p=p[np.argsort(-p[:,4],kind='stable')]
        ov=overlaps(p,g);iof=overlaps(p,ann['bboxes_ignore'],True);used=set();tp=fp=ig=0;matches=[]
        for k,b in enumerate(p):
            avail=[j for j in range(len(g)) if j not in used]
            best=max(avail,key=lambda j:ov[k,j]) if avail else None
            if best is not None and ov[k,best]>=.5:
                used.add(best);tp+=1;matches.append(dict(pred=k,status='TP',gt=best,iou=float(ov[k,best])))
            elif iof.shape[1] and iof[k].max()>.5:
                ig+=1;matches.append(dict(pred=k,status='IGNORE'))
            else:fp+=1;matches.append(dict(pred=k,status='FP'))
        rows.append(dict(category=CLASSES[cid],TP=tp,FP=fp,FN=len(g)-tp,GT_count=len(g),pseudo_count=len(p),ignored_pseudo_count=ig))
        details.append(matches)
    return rows,details

def ratios(r,nframes):
    r=dict(r);tp,fp,fn=r['TP'],r['FP'],r['FN']
    r.update(precision_ratio=tp/(tp+fp) if tp+fp else 'N/A',recall_ratio=tp/(tp+fn) if tp+fn else 'N/A',miss_rate_ratio=fn/(tp+fn) if tp+fn else 'N/A',TP_per_frame=tp/nframes,FP_per_frame=fp/nframes,pseudo_count_per_frame=r['pseudo_count']/nframes,frames=nframes)
    return r

def main():
    p=argparse.ArgumentParser();p.add_argument('--timepoint',choices=['e0','e1','e3','e5'],required=True);p.add_argument('--smoke',action='store_true');args=p.parse_args()
    label={'e0':'F_stage3_e5','e1':'F_stage4_e1','e3':'F_stage4_e3','e5':'F_stage4_e5'}[args.timepoint]
    root=RUN/'results/exp3'/args.timepoint;root.mkdir(parents=True,exist_ok=True)
    dump(root/'job.json',dict(pid=os.getpid(),gpu=os.environ.get('CUDA_VISIBLE_DEVICES'),argv=sys.argv,start=time.time()))
    fixed_seed();torch.backends.cudnn.benchmark=True;ds=dataset('night');assert len(ds)==7286
    teacher,cfg=load_teacher(label)
    from darth.models.utils.transforms import detections2bboxes,filter_bboxes_by_confidence
    videos=defaultdict(list)
    for info in ds.data_infos:videos[info['video_id']].append(info)
    start=time.time()
    with torch.no_grad():
        for v,infos in videos.items():
            path=root/'predictions'/('%s.pkl'%v)
            if path.exists():
                with open(path,'rb') as f:old=pickle.load(f)
                assert old['image_ids']==[im['id'] for im in infos]
                continue
            if args.smoke:infos=infos[:2]
            loader=torch.utils.data.DataLoader(NightFrames(infos),batch_size=1,num_workers=2,shuffle=False,collate_fn=identity)
            predictions=[];stats=[];matches=[]
            for info,(x,meta) in zip(infos,loader):
                x=x.cuda();feat=teacher.extract_feat(x)
                proposals=teacher.rpn_head.simple_test_rpn(feat,meta)
                detections=teacher.roi_head.simple_test(feat,proposals,meta,rescale=False)
                boxes,labels,scores=detections2bboxes(detections)
                boxes,labels,scores=filter_bboxes_by_confidence(boxes,labels,scores,conf_thr=cfg.model.conf_thr)
                # Teacher boxes are before teacher_pipeline geometry. Invert only
                # base resize to original coordinates, matching dataset GT.
                boxes=boxes[0]/meta[0]['scale_factor'];labels=labels[0];scores=scores[0]
                pred=[np.concatenate((boxes[labels==c],scores[labels==c,None]),axis=1) for c in range(6)]
                ann=ds.get_ann_info(info);counts,mt=match_image(pred,ann)
                predictions.append(pred);stats.append(counts);matches.append(mt)
            if args.smoke:
                print('SMOKE',args.timepoint,len(predictions),stats,flush=True);return
            save_pickle(path,dict(image_ids=[im['id'] for im in infos],predictions=predictions,statistics=stats,matches=matches))
            print('video_complete',args.timepoint,v,len(infos),'elapsed',time.time()-start,flush=True)
    keys=['TP','FP','FN','GT_count','pseudo_count','ignored_pseudo_count'];totals={c:{k:0 for k in keys} for c in CLASSES};pervideo=[];nframes=0
    for v,infos in videos.items():
        with open(root/'predictions'/('%s.pkl'%v),'rb') as f:r=pickle.load(f)
        assert r['image_ids']==[im['id'] for im in infos] and len(r['statistics'])==len(infos)
        vt={c:{k:0 for k in keys} for c in CLASSES}
        for counts in r['statistics']:
            for row in counts:
                for k in keys:vt[row['category']][k]+=row[k]
        for c in CLASSES:
            for k in keys:totals[c][k]+=vt[c][k]
            pervideo.append(ratios(dict(timepoint=args.timepoint,checkpoint=label,video_id=v,category=c,**vt[c]),len(infos)))
        micro={k:sum(vt[c][k] for c in CLASSES) for k in keys}
        pervideo.append(ratios(dict(timepoint=args.timepoint,checkpoint=label,video_id=v,category='micro_six_classes',**micro),len(infos)))
        nframes+=len(infos)
    assert nframes==7286 and len(videos)==36
    perclass=[ratios(dict(timepoint=args.timepoint,checkpoint=label,category=c,**totals[c]),nframes) for c in CLASSES]
    micro={k:sum(totals[c][k] for c in CLASSES) for k in keys}
    summary=[ratios(dict(timepoint=args.timepoint,checkpoint=label,category='micro_six_classes',**micro),nframes),next(r for r in perclass if r['category']=='car')]
    write_csv(root/'quality_summary.csv',summary);write_csv(root/'quality_per_class.csv',perclass);write_csv(root/'quality_per_video.csv',pervideo)
    dump(root/'complete.json',dict(status='complete',frames=nframes,videos=len(videos),exit_code=0,branch='teacher_detector',time=time.time()))
    print('COMPLETE',args.timepoint,summary,flush=True)

if __name__=='__main__':main()
