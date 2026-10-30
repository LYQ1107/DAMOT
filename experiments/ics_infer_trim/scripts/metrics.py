"""Native full-domain metrics and exactly matched held-out-video statistics."""
import argparse,pickle,shutil
from collections import defaultdict
from common import *

def cached_hota(source):
    import trackeval
    h=trackeval.metrics.HOTA({'PRINT_CONFIG':False});classes={}
    for c in CLASSES:
        result={}
        for r in rows(source/'evaluation/track'/(c+'_detailed.csv')):
            result[r['seq']]={k:(np.array([float(r[k+'___'+str(int(round(a*100)))]) for a in h.array_labels]) if k+'___5' in r else float(r[k])) for k in h.fields}
        combined=h.combine_sequences({k:v for k,v in result.items() if k!='COMBINED'})
        assert all(np.allclose(combined[k],result['COMBINED'][k],atol=1e-10) for k in h.fields)
        classes[c]=result
    names=set(classes['car'])-{'COMBINED'}
    whole=h.combine_classes_det_averaged({c:classes[c]['COMBINED'] for c in CLASSES})
    per={v:h.combine_classes_det_averaged({c:classes[c][v] for c in CLASSES}) for v in names}
    return classes,per,whole

def main():
    p=argparse.ArgumentParser();p.add_argument('--label',required=True);p.add_argument('--domain',required=True);p.add_argument('--source');a=p.parse_args()
    import torch
    from mmcv import Config
    from darth.datasets import build_dataset
    from darth.datasets.eval.coco_eval_dataset import BDD100K
    from mmtrack.core.evaluation.eval_mot import acc_single_video
    import motmetrics as mm
    torch.set_num_threads(1)
    out=ROOT/'results'/a.label/a.domain
    if (out/'metrics_complete.json').exists():return
    cfg=Config.fromfile(str(out/'used_config.py'));ds=build_dataset(cfg.data.test)
    if a.label.endswith('_b0'):
        original=Path(a.source) if a.source else CROOT/'evaluation'/a.label.split('_')[0]/a.domain
        for name in ['summary_metrics.csv','track_metrics.csv','hota_metrics.csv']:shutil.copyfile(str(original/name),str(out/name))
        ev=out/'evaluation';ev.mkdir(exist_ok=True)
        if not (ev/'track').exists():(ev/'track').symlink_to(original/'evaluation/track',target_is_directory=True)
    else:
        def all_frames(self):
            by={v['id']:[] for v in self.gt_data['videos']}
            for im in self.gt_data['images']:by[im['video_id']].append(im)
            return {v:{im['id']:i for i,im in enumerate(sorted(ims,key=lambda x:x['frame_id']))} for v,ims in by.items()}
        BDD100K._compute_image_to_timestep_mappings=all_frames
        if not (out/'summary_metrics.csv').exists():
            if not (out/'evaluation/summary_metrics.csv').exists():
                with (out/'raw_results.pkl').open('rb') as f:pred=pickle.load(f)
                ds.evaluate(pred,metric=['track'],results_path=str(out/'evaluation'),
                     results_kwargs=dict(scalabel=False,append_empty_preds=True),
                     track_kwargs=dict(iou_thr=.5,ignore_iof_thr=.5,ignore_by_classes=False,nproc=1,majority_voting=False))
            for name in ['summary_metrics.csv','track_metrics.csv','hota_metrics.csv']:shutil.copyfile(str(out/'evaluation'/name),str(out/name))
    # Native optional metrics (e.g. MOTP for a class with no matches) may be NaN.
    # Keep the native CSVs intact; encode only these undefined JSON values as null.
    flat={};undefined=[]
    for kind in ['summary','track','hota']:
        for r in rows(out/(kind+'_metrics.csv')):
            for k,v in r.items():
                if k=='':continue
                key=kind+'/'+r['']+'/'+k;value=float(v) if v else float('nan')
                flat[key]=value if np.isfinite(value) else None
                if flat[key] is None:undefined.append(key)
    dump(out/'evaluation_flat_metrics.json',dict(source='Native CSVs, including native summary rows',values=flat,undefined_values_encoded_as_null=undefined))
    summary={r['']:r for r in rows(out/'summary_metrics.csv')};track={r['']:r for r in rows(out/'track_metrics.csv')}
    aggregated=[]
    for c in CLASSES+['OVERALL']:
        r=dict(model=a.label,domain=a.domain,category=c)
        for k in ['HOTA','DetA','IDF1','AssA']:r[k]=float(summary[c][k])
        r['IDSW']=int(float(track[c]['IDSw']));aggregated.append(r)
    csvout(out/'metrics.csv',aggregated)
    ch,per,whole=cached_hota(out);videos=defaultdict(list)
    for i,im in enumerate(ds.data_infos):videos[im['video_id']].append(i)
    names={v['id']:v['name'] for v in ds.coco.dataset['videos']}
    sorted_names=sorted(set(names.values()));fit={v for v in sorted_names if int(hashlib.md5((v+'/0').encode()).hexdigest()[:8],16)%2==0}
    if len(fit) in (0,len(sorted_names)):fit=set(sorted_names[::2])
    mh=mm.metrics.create();pervideo=[];counts={c:[] for c in ['OVERALL','car']}
    # Reuse already validated frozen ID accumulators for beta0, avoiding a redundant pass.
    old=ADV/'tracking'/a.label.split('_')[0]/a.domain/'per_video.csv'
    original_ids={r['video']+'|'+r['category']:r for r in rows(old)} if a.label.endswith('_b0') and old.exists() else None
    pred=None
    if original_ids is None:
        with (out/'raw_results.pkl').open('rb') as f:pred=pickle.load(f)
    for vid,ids in videos.items():
        if original_ids is None:
            anns=[ds.get_ann_info(ds.data_infos[i]) for i in ids]
            accs=acc_single_video([pred['track_bboxes'][i] for i in ids],anns,.5,.5,False)
            s=mh.compute_many(accs,names=CLASSES,metrics=['idf1','idtp','idfp','idfn','num_switches'],generate_overall=True)
        for c in ['OVERALL','car']:
            hc=per[names[vid]] if c=='OVERALL' else ch[c][names[vid]]
            if original_ids is None:
                val=s.loc[c];q={k:float(val[k]) for k in ['idtp','idfp','idfn','num_switches']};idf=100*float(val['idf1']) if np.isfinite(val['idf1']) else 0.
            else:
                prev=original_ids[names[vid]+'|'+c];q=dict(idtp=float(prev['IDTP']),idfp=float(prev['IDFP']),idfn=float(prev['IDFN']),num_switches=float(prev['IDSW']));idf=float(prev['IDF1'])
            counts[c].append(q)
            pervideo.append(dict(model=a.label,domain=a.domain,video=names[vid],video_id=vid,category=c,test=names[vid] not in fit,frames=len(ids),
                 HOTA=100*float(np.mean(hc['HOTA'])),DetA=100*float(np.mean(hc['DetA'])),AssA=100*float(np.mean(hc['AssA'])),
                 IDF1=idf,IDSW=int(q['num_switches']),IDTP=q['idtp'],IDFP=q['idfp'],IDFN=q['idfn']))
    parity=[]
    for c in ['OVERALL','car']:
        hc=whole if c=='OVERALL' else ch[c]['COMBINED'];total={k:sum(v[k] for v in counts[c]) for k in ['idtp','idfp','idfn','num_switches']}
        got={k:100*float(np.mean(hc[k])) for k in ['HOTA','DetA','AssA']}
        got.update(IDF1=200*total['idtp']/max(1,2*total['idtp']+total['idfp']+total['idfn']),IDSW=total['num_switches'])
        ref=next(r for r in aggregated if r['category']==c)
        for k,v in got.items():parity.append(dict(category=c,metric=k,observed=v,reference=ref[k],delta=v-ref[k],passed=abs(v-ref[k])<1e-6))
    csvout(out/'per_video.csv',pervideo);csvout(out/'aggregate_parity.csv',parity)
    assert all(r['passed'] for r in parity),parity
    dump(out/'metrics_complete.json',dict(status='complete',frames=len(ds),videos=len(videos),test_videos=sum(v not in fit for v in names.values()),
         all_frame_hota=True,aggregate_parity=True,original_prediction_sha256=sha(out/'raw_results.pkl')))
    print('METRICS_COMPLETE',a.label,a.domain,flush=True)

if __name__=='__main__':main()
