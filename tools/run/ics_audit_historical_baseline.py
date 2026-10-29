"""CPU audit of native beta0 and historical prediction evaluation; no intervention."""
import csv,json,sys
from pathlib import Path
import pandas as pd
import trackeval
import ics_tracking_intervention as s4
from apr_complement_experiment import np,build_dataset,dump,sha,Config
from evaluate_apr_complement import all_frame_hota_index,csv_write
from darth.datasets.eval.coco_eval_dataset import BDD100K
from darth.datasets.eval.coco_eval_utils import get_default_trackeval_config,get_default_trackeval_dataset_config
from darth.datasets.eval.logging_utils import accumulate_results
ROOT=Path(__file__).resolve().parents[3];OLD=Path('/data1/liuyeqiang/damot_rebuttal_20261026')

def historical():
    manifest=json.loads((OLD/'manifests/all_frame_hota_complete.json').read_text())
    job=next(r for r in manifest['jobs'] if r['kind']=='exp1' and r['model']=='F' and r['stage']==4 and r['domain']=='overcast')
    out=ROOT/'results/S4/F/historical_prediction_recheck';out.mkdir(parents=True,exist_ok=True)
    ec=get_default_trackeval_config();ec.update(USE_PARALLEL=False,PRINT_RESULTS=False,TIME_PROGRESS=False,OUTPUT_SUMMARY=True,OUTPUT_DETAILED=True,PLOT_CURVES=False)
    dc=get_default_trackeval_dataset_config();dc.update(CLASSES_TO_EVAL=list(s4.CLASSES),TRACKERS_TO_EVAL=['track'],OUTPUT_FOLDER=str(out),GT_FOLDER=str(OLD/'data/annotations/overcast.json'),TRACKER_SUB_FOLDER=job['prediction'])
    ds=BDD100K(dc);vals,_=trackeval.Evaluator(ec).evaluate([ds],[trackeval.metrics.HOTA({'PRINT_CONFIG':False})])
    block=vals[ds.get_name()]['track']['COMBINED_SEQ'];metrics,_=accumulate_results(block,['HOTA','DetA','AssA']);pd.DataFrame.from_dict(metrics).to_csv(out/'hota_metrics.csv')
    calculated={r['']:r for r in csv.DictReader((out/'hota_metrics.csv').open())};comparison=[]
    for category in ['car','OVERALL']:
        ref=s4.archived_expected('F','overcast')[category]
        for m in ['HOTA','DetA','AssA']:
            val=float(calculated[category][m]);base=float(ref[m+'_percent']);assert abs(val-base)<1e-9,(category,m,val,base)
            comparison.append(dict(category=category,metric=m,rechecked=val,archived=base,difference=val-base))
    csv_write(out/'comparison.csv',comparison);dump(out/'complete.json',dict(status='PASS',prediction=job['prediction'],prediction_sha256=sha(job['prediction']),current_evaluator_reproduces_frozen_metrics=True))
    print('HISTORICAL PREDICTION RECHECK PASS',flush=True)

def audit(domain):
    out=ROOT/'results/S4/F/baseline_audit'/domain;out.mkdir(parents=True,exist_ok=True)
    if (out/'complete.json').exists():return
    cfg=s4.config(domain);ds=build_dataset(cfg.data.test);vids={}
    for i,im in enumerate(ds.data_infos):vids.setdefault(im['video_id'],[]).append(i)
    results=dict(det_bboxes=[],track_bboxes=[])
    for v,indices in vids.items():
        recs=s4.read(ROOT/'cache/S4/F'/domain/('video_%s.pkl.gz'%v));assert [r['index'] for r in recs]==indices
        for r in recs:
            for k in results:results[k].append(r['output'][k])
    ds.evaluate(results,metric=['track'],results_path=str(out/'evaluation'),results_kwargs=dict(scalabel=False,append_empty_preds=True),track_kwargs=dict(iou_thr=.5,ignore_iof_thr=.5,ignore_by_classes=False,nproc=1,majority_voting=False))
    reference=s4.archived_expected('F',domain);rows=[]
    for r in csv.DictReader((out/'evaluation/summary_metrics.csv').open()):
        if r[''] not in ['car','OVERALL']:continue
        row=dict(checkpoint='F_stage4_e5',domain=domain,category=r[''],frames=len(ds))
        for m in ['HOTA','DetA','AssA','IDF1']:
            val=float(r[m]);row[m+'_percent']=val;row['archived_'+m+'_percent']=float(reference[r['']][m+'_percent']);row['difference_'+m+'_pp']=val-row['archived_'+m+'_percent']
        rows.append(row)
    csv_write(out/'metrics.csv',rows);dump(out/'complete.json',dict(status='audit_complete',fresh_baseline_not_yet_authorized=True,nonzero_beta_executed=False))
    print('NATIVE BETA0 AUDIT',domain,rows,flush=True)

def main():
    BDD100K._compute_image_to_timestep_mappings=all_frame_hota_index
    historical()
    for d in ['overcast','rainy','dawn_dusk','night']:audit(d)
if __name__=='__main__':main()
