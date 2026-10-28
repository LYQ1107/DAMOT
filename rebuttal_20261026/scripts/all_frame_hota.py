#!/usr/bin/env python3
"""Re-evaluate qualified predictions; include GT-empty images in HOTA indexing."""
import os,time,json,subprocess,csv,shutil,hashlib,concurrent.futures
from pathlib import Path
from runtime import *
from bulk_cache_connection import SSH
from darth.datasets.eval.coco_eval_dataset import BDD100K
from darth.datasets.eval.coco_eval_utils import get_default_trackeval_config,get_default_trackeval_dataset_config
from darth.datasets.eval.logging_utils import accumulate_results
R=Path('/data1/liuyeqiang/nas/A100/data1/LWR/vranlee/SERVER_ONLY/avis/DA-MOT/darth-main')
class AllFrameBDD100K(BDD100K):
    def _compute_image_to_timestep_mappings(self):
        by={v['id']:[] for v in self.gt_data['videos']}
        for im in self.gt_data['images']:by[im['video_id']].append(im)
        return {v:{im['id']:i for i,im in enumerate(sorted(ims,key=lambda x:x['frame_id']))} for v,ims in by.items()}

def cache_historical():
    rows=json.loads((RUN/'results/exp1/reuse_provenance.json').read_text());cache=RUN/'cache/historical_predictions';cache.mkdir(parents=True,exist_ok=True)
    listing=RUN/'manifests/historical_prediction_tar_filelist.txt'
    files=[str(Path(r['source']).with_name('track_coco.json').relative_to(R)) for r in rows]
    missing=[s for s in files if not (cache/s).exists()]
    if missing:
        listing.write_text(''.join(s+'\n' for s in missing));remote='/home/Valadmin/A100/data1/LWR/vranlee/SERVER_ONLY/avis/DA-MOT/darth-main'
        with open(listing,'rb') as f:
            sender=subprocess.Popen(SSH+['tar -C '+remote+' -cf - -T -'],stdin=f,stdout=subprocess.PIPE)
            receiver=subprocess.Popen(['tar','-xf','-','-C',str(cache)],stdin=sender.stdout);sender.stdout.close()
            assert receiver.wait()==0 and sender.wait()==0
    jobs=[]
    for row,s in zip(rows,files):
        jobs.append(dict(kind='exp1',model=row['model'],stage=row['stage'],domain=row['domain'],prediction=str(cache/s),output=str(RUN/'results/exp1/all_frame_hota'/row['model']/('stage%d'%row['stage'])/row['domain'])))
    return jobs

def evaluate(job):
    out=Path(job['output']);complete=out/'complete.json'
    if complete.exists():return json.loads(complete.read_text())
    out.mkdir(parents=True,exist_ok=True)
    ann=RUN/'data/annotations'/(job['domain']+'.json');pred=Path(job['prediction'])
    ec=get_default_trackeval_config();ec.update(USE_PARALLEL=False,PRINT_RESULTS=False,TIME_PROGRESS=False,OUTPUT_SUMMARY=True,OUTPUT_DETAILED=True,PLOT_CURVES=False)
    dc=get_default_trackeval_dataset_config();dc.update(CLASSES_TO_EVAL=list(CLASSES),TRACKERS_TO_EVAL=['track'],OUTPUT_FOLDER=str(out),GT_FOLDER=str(ann),TRACKER_SUB_FOLDER=str(pred))
    import trackeval,pandas as pd
    ds=AllFrameBDD100K(dc)
    assert all(len(ds.seq2images2timestep[v])==ds.seq_lengths[v] for v in ds.seq_lengths)
    vals,_=trackeval.Evaluator(ec).evaluate([ds],[trackeval.metrics.HOTA({'PRINT_CONFIG':False})])
    block=vals[ds.get_name()]['track']['COMBINED_SEQ']
    metrics,_=accumulate_results(block,['HOTA','DetA','AssA','DetRe','DetPr','AssRe','AssPr','LocA','Dets','IDs','GT_Dets'])
    pd.DataFrame.from_dict(metrics).to_csv(out/'hota_metrics.csv')
    save_pickle(out/'raw_trackeval_results.pkl',vals)
    rec=dict(**job,status='complete',frames=sum(ds.seq_lengths.values()),videos=len(ds.seq_list),exit_code=0,all_frames_in_time_index=True,prediction_sha256=hashlib.sha256(pred.read_bytes()).hexdigest())
    dump(complete,rec);print('HOTA COMPLETE',job['kind'],job['model'],job['stage'],job['domain'],flush=True)
    return rec

def readcsv(p):
    with open(p) as f:return list(csv.DictReader(f))
def csvmap(p):return {r['']:r for r in readcsv(p)}
def overlay_exp1():
    root=RUN/'results/exp1';p=root/'stage_domain_metrics.csv'
    if not (root/'stage_domain_metrics_legacy_eval.csv').exists():shutil.copyfile(p,root/'stage_domain_metrics_legacy_eval.csv')
    rows=readcsv(root/'stage_domain_metrics_legacy_eval.csv')
    for r in rows:
        out=root/'all_frame_hota'/r['model']/('stage'+r['stage'])/r['eval_domain']
        vals=csvmap(out/'hota_metrics.csv')[r['category']]
        for name in ['HOTA','DetA','AssA']:
            r[name+'_legacy_percent']=r[name+'_percent'];r[name+'_percent']=float(vals[name])
        r['hota_provenance']='re-evaluated existing qualified predictions; fixed all-frame HOTA index'
        r['all_frame_hota_source']=str(out/'hota_metrics.csv')
    write_csv(p,rows)
    old=root/'apr_old_domain_metrics.csv'
    if not (root/'apr_old_domain_metrics_legacy_eval.csv').exists():shutil.copyfile(old,root/'apr_old_domain_metrics_legacy_eval.csv')
    write_csv(old,[r for r in rows if r['stage']=='4' and r['eval_domain']!='night'])
    f=root/'apr_forgetting_metrics.csv'
    if not (root/'apr_forgetting_metrics_legacy_eval.csv').exists():shutil.copyfile(f,root/'apr_forgetting_metrics_legacy_eval.csv')
    output=[];doms=['overcast','rainy','dawn_dusk']
    for m in ['N_old','F_old','F']:
        for cls in ['OVERALL','car']:
            for metric in ['HOTA_percent','IDF1_percent','AssA_percent']:
                cells={(int(r['stage']),r['eval_domain']):float(r[metric]) for r in rows if r['model']==m and r['category']==cls}
                fs=[max(cells[i,doms[j-1]] for i in range(j,4))-cells[4,doms[j-1]] for j in range(1,4)]
                output.append(dict(model=m,category=cls,metric=metric,FinalOld_percent=sum(cells[4,d] for d in doms)/3,AvgF_percentage_points=sum(fs)/3,F_overcast_pp=fs[0],F_rainy_pp=fs[1],F_dawn_dusk_pp=fs[2],formula='mean_j=1..3[max_i=j..3 R_ij - R_4j]; no clipping',provenance='IDF1 reused full precision; HOTA/AssA re-evaluated all frames from existing predictions'))
    write_csv(f,output)
    with open(root/'comparison_config_notes.txt','a') as f:f.write('\nAll-frame HOTA correction: archived evaluator omitted predictions on images with no GT annotations from its timestep map. Re-evaluated all 48 qualified prediction JSONs with every annotation image indexed; HOTA/DetA/AssA updated consistently across models/stages. Legacy CSVs remain under *_legacy_eval.csv; CLEAR IDF1/IDSw already included such frames and are unchanged. No new model forward.\n')

def overlay_exp2(jobs):
    import pandas as pd
    for j in jobs:
        root=RUN/'results/exp2'/j['model'];new=Path(j['output'])/'hota_metrics.csv'
        for k in ['hota','summary']:
            p=root/(k+'_metrics.csv');legacy=root/(k+'_metrics_legacy_eval.csv')
            if not legacy.exists():shutil.copyfile(p,legacy)
        shutil.copyfile(new,root/'hota_metrics.csv')
        summ=pd.read_csv(root/'summary_metrics_legacy_eval.csv',index_col=0);h=pd.read_csv(new,index_col=0)
        for k in ['HOTA','DetA','AssA','LocA']:summ[k]=h[k]
        summ.to_csv(root/'summary_metrics.csv')

if __name__=='__main__':
    start=time.time();jobs=cache_historical()
    with concurrent.futures.ProcessPoolExecutor(max_workers=4) as pool:records=list(pool.map(evaluate,jobs))
    overlay_exp1();jobs2=[]
    for i,m in enumerate(['S','F_stage1_e5','F_stage2_e5','F_stage3_e5','F_stage4_e5']):
        root=RUN/'results/exp2'/m
        while not (root/'complete.json').exists():time.sleep(15)
        jobs2.append(dict(kind='exp2',model=m,stage=i,domain='clear_holdout_45v',prediction=str(root/'evaluation/track_coco.json'),output=str(root/'evaluation_all_frames')))
    with concurrent.futures.ProcessPoolExecutor(max_workers=4) as pool:records+=list(pool.map(evaluate,jobs2))
    overlay_exp2(jobs2)
    dump(RUN/'manifests/all_frame_hota_complete.json',dict(status='complete',exit_code=0,jobs=records,seconds=time.time()-start,correction='Index all GT annotation images, including empty-GT frames; otherwise original evaluator and aggregation unchanged'))
    print('ALL FRAME HOTA COMPLETE',len(records),flush=True)
