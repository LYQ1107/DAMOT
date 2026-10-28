#!/usr/bin/env python3
import argparse,os,time,json,pickle
from functools import partial
from runtime import *

def main():
    p=argparse.ArgumentParser();p.add_argument('--model',required=True);p.add_argument('--smoke',action='store_true');p.add_argument('--evaluate-only',action='store_true');args=p.parse_args()
    fixed_seed();torch.backends.cudnn.benchmark=True
    root=RUN/'results/exp2'/args.model;root.mkdir(parents=True,exist_ok=True)
    dump(root/'job.json',dict(pid=os.getpid(),gpu=os.environ.get('CUDA_VISIBLE_DEVICES'),argv=sys.argv,start=time.time()))
    ds=dataset('clear_holdout_45v')
    vids={}
    for idx,im in enumerate(ds.data_infos):vids.setdefault(im['video_id'],[]).append(idx)
    expected=9103 if not args.smoke else 2
    assert len(ds)==9103 and len(vids)==45
    if not args.evaluate_only:
        model,cfg=load_model(args.model,deploy=True,fusion=True)
        from torch.utils.data import DataLoader,Subset
        from mmcv.parallel import collate,scatter
        inds=[];complete=set()
        for v,ixs in vids.items():
            path=root/'predictions'/('%s.pkl'%v)
            if path.exists():
                with open(path,'rb') as f:r=pickle.load(f)
                assert r['indices']==ixs and len(r['outputs'])==len(ixs)
                complete.add(v)
            else:inds.extend(ixs)
        if args.smoke:inds=inds[:2]
        loader=DataLoader(Subset(ds,inds),batch_size=1,shuffle=False,num_workers=4,collate_fn=partial(collate,samples_per_gpu=1),pin_memory=False)
        outputs=[];active=None;start=time.time()
        with torch.no_grad():
            for n,(idx,data) in enumerate(zip(inds,loader)):
                im=ds.data_infos[idx];v=im['video_id']
                if active!=v:
                    assert im['frame_id']==0
                    model.tracker.reset();active=v;outputs=[]
                batch=scatter(data,[0])[0]
                output=model(return_loss=False,rescale=True,**batch)
                outputs.append(output)
                if idx==vids[v][-1]:
                    save_pickle(root/'predictions'/('%s.pkl'%v),dict(indices=vids[v],outputs=outputs,video_id=v))
                    print('video_complete',args.model,v,len(outputs),'seconds',time.time()-start,flush=True)
                if args.smoke:print('smoke',idx,{k:[len(x) for x in val] for k,val in output.items()},flush=True)
        if args.smoke:return
        del model;torch.cuda.empty_cache()
    results={'det_bboxes':[],'track_bboxes':[]};seen=[]
    for v,ixs in vids.items():
        with open(root/'predictions'/('%s.pkl'%v),'rb') as f:r=pickle.load(f)
        assert r['indices']==ixs
        seen+=ixs
        for o in r['outputs']:
            for k in results:results[k].append(o[k])
    assert len(seen)==len(set(seen))==9103 and seen==list(range(9103))
    save_pickle(root/'raw_results.pkl',results)
    # Original dataset evaluator; plots explicitly disabled in the local copy.
    metrics=ds.evaluate(results,metric=['track'],results_path=str(root/'evaluation'),results_kwargs=dict(scalabel=False,append_empty_preds=True),track_kwargs=dict(iou_thr=.5,ignore_iof_thr=.5,ignore_by_classes=False,nproc=1,majority_voting=False))
    dump(root/'evaluation_flat_metrics.json',metrics)
    import shutil
    for k in ['track','hota','summary']:
        shutil.copyfile(root/'evaluation'/(k+'_metrics.csv'),root/(k+'_metrics.csv'))
    # Per-video CLEAR metrics use precisely the same official accumulator.
    from mmtrack.core.evaluation.eval_mot import acc_single_video,eval_single_class,METRIC_MAPS
    from darth.datasets.eval.logging_utils import rename_dict
    pervideo=[]
    for v,ixs in vids.items():
        ann=[ds.get_ann_info(ds.data_infos[i]) for i in ixs]
        accs=acc_single_video([results['track_bboxes'][i] for i in ixs],ann,.5,.5,False)
        for cid,a in enumerate(accs):
            if a is not None:
                vals=eval_single_class([str(v)],[a]);row=dict(video_id=v,category=CLASSES[cid])
                row.update({METRIC_MAPS[k]:val for k,val in zip(METRIC_MAPS,vals)});pervideo.append(row)
    write_csv(root/'clear_metrics_per_video.csv',pervideo)
    dump(root/'complete.json',dict(status='complete',frames=9103,videos=45,exit_code=0,model=args.model,time=time.time()))
    print('COMPLETE',args.model,flush=True)

if __name__=='__main__':main()
