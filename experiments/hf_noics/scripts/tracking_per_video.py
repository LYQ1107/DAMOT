"""Reuse frozen TrackEval arrays and native MOTMetrics IDs, proving aggregate parity."""
import argparse,json,pickle
from collections import defaultdict
from common import *

def hota_cached(source):
    import trackeval
    h=trackeval.metrics.HOTA({'PRINT_CONFIG':False});classes={}
    for c in CLASSES:
        file=source/'evaluation/track'/(c+'_detailed.csv')
        rr=readcsv(file);out={}
        for r in rr:
            result={}
            for key in h.fields:
                if key+'___5' in r:
                    result[key]=np.array([float(r[key+'___'+str(int(round(a*100)))]) for a in h.array_labels])
                else:result[key]=float(r[key])
            out[r['seq']]=result
        classes[c]=out
        recomputed=h.combine_sequences({k:v for k,v in out.items() if k!='COMBINED'})
        for k in h.fields:assert np.allclose(recomputed[k],out['COMBINED'][k],atol=1e-10),(source,c,k)
    sequences=sorted(set(classes['car'])-{'COMBINED'})
    combined={v:h.combine_classes_det_averaged({c:classes[c][v] for c in CLASSES}) for v in sequences}
    overall=h.combine_classes_det_averaged({c:classes[c]['COMBINED'] for c in CLASSES})
    return h,classes,combined,overall

def main(label,domain,source=None):
    source=Path(source) if source else CROOT/'evaluation'/label/domain
    target=ROOT/'tracking'/label/domain
    if (target/'complete.json').exists():return
    import torch
    torch.set_num_threads(1)
    import darth.datasets
    from darth.datasets import build_dataset
    from mmcv import Config
    from mmtrack.core.evaluation.eval_mot import acc_single_video
    import motmetrics as mm
    cfg=Config.fromfile(str(source/'used_config.py'))
    ds=build_dataset(cfg.data.test)
    with (source/'raw_results.pkl').open('rb') as f:pred=pickle.load(f)
    assert len(pred['track_bboxes'])==len(ds)
    h,classes,hs,ha=hota_cached(source)
    videos=defaultdict(list)
    for i,im in enumerate(ds.data_infos):videos[im['video_id']].append(i)
    names={v['id']:v['name'] for v in ds.coco.dataset['videos']}
    fit=dict(zip(sorted(names),probe.split_mask(np.array([names[v] for v in sorted(names)]),0)))
    mh=mm.metrics.create();rows=[];counts={c:[] for c in ['OVERALL','car']}
    for vid,ids in videos.items():
        anns=[ds.get_ann_info(ds.data_infos[i]) for i in ids]
        accs=acc_single_video([pred['track_bboxes'][i] for i in ids],anns,.5,.5,False)
        summary=mh.compute_many(accs,names=CLASSES,metrics=['idf1','idtp','idfp','idfn','num_switches'],generate_overall=True)
        for c in ['OVERALL','car']:
            values=summary.loc[c];hc=hs[names[vid]] if c=='OVERALL' else classes[c][names[vid]]
            q={k:float(values[k]) for k in ['idtp','idfp','idfn','num_switches']}
            counts[c].append(q)
            rows.append(dict(model=label,domain=domain,video=names[vid],video_id=vid,category=c,test=not bool(fit[vid]),frames=len(ids),
                       HOTA=100*float(np.mean(hc['HOTA'])),AssA=100*float(np.mean(hc['AssA'])),
                       IDF1=100*float(values['idf1']) if np.isfinite(values['idf1']) else 0.,IDSW=int(q['num_switches']),
                       IDTP=q['idtp'],IDFP=q['idfp'],IDFN=q['idfn']))
        print('VIDEO_METRICS',label,domain,names[vid],flush=True)
    old={r['']:r for r in readcsv(source/'summary_metrics.csv')}
    old_ids={r['']:r for r in readcsv(source/'track_metrics.csv')};parity=[]
    for c in ['OVERALL','car']:
        hh=ha if c=='OVERALL' else classes[c]['COMBINED']
        totals={k:sum(v[k] for v in counts[c]) for k in ['idtp','idfp','idfn','num_switches']}
        got=dict(HOTA=100*float(np.mean(hh['HOTA'])),AssA=100*float(np.mean(hh['AssA'])),
                 IDF1=200*totals['idtp']/max(1,2*totals['idtp']+totals['idfp']+totals['idfn']),IDSW=totals['num_switches'])
        for metric,value in got.items():
            expected=float(old_ids[c]['IDSw']) if metric=='IDSW' else float(old[c][metric])
            parity.append(dict(model=label,domain=domain,category=c,metric=metric,observed=value,reference=expected,delta=value-expected,passed=abs(value-expected)<1e-6))
    csvout(target/'per_video.csv',rows);csvout(target/'aggregate_parity.csv',parity)
    assert all(r['passed'] for r in parity),('Original aggregate mismatch',parity)
    dump(target/'complete.json',dict(status='complete',source=str(source),videos=len(videos),frames=len(ds),
         raw_predictions_sha256=sha(source/'raw_results.pkl'),annotation_sha256=sha(cfg.data.test.ann_file),
         HOTA_AssA='Frozen original TrackEval detailed arrays, checked through native combine_sequences and combine_classes_det_averaged',
         IDF1_IDSW='Same MMTracking/MOTMetrics accumulators as original evaluator; per-video identity counts sum to original aggregate exactly',
         note='Native evaluator already mixes TrackEval HOTA/AssA with MOTMetrics IDF1/IDSW; keeping this protocol fulfills aggregate parity. Not switching ID matching implementations.'))
    print('AGGREGATE_PARITY_PASS',label,domain,flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--model',required=True);p.add_argument('--domain',required=True);p.add_argument('--source');a=p.parse_args()
    main(a.model,a.domain,a.source)
