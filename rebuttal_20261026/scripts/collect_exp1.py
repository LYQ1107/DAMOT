#!/usr/bin/env python3
import csv,json,hashlib,shutil,re
from pathlib import Path
from collections import Counter
from prepare_inputs import RUN,R,F,CLASSES,STAGES,dump,sha

def readcsv(p):
    with open(p) as f:return {r['']:r for r in csv.DictReader(f)}
def writecsv(p,rows):
    with open(p,'w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
def gt_counts(domain):
    p=RUN/'data/annotations'/(domain+'.json')
    if not p.exists():p=R/('data/bdd_da/annotations/box_track_20/box_track_%s_all_cocofmt.json'%domain)
    a=json.loads(p.read_text()); ims={i['id']:i for i in a['images']}; cats={c['id']:c['name'] for c in a['categories']};c=Counter()
    for b in a['annotations']:
        if b.get('ignore',False) or b.get('iscrowd',False):continue
        im=ims[b['image_id']];x,y,w,h=b['bbox']
        if min(x+w,im['width'])-max(x,0)<=0 or min(y+h,im['height'])-max(y,0)<=0:continue
        if b['area']<=0 or w<1 or h<1 or cats[b['category_id']] not in CLASSES:continue
        c[cats[b['category_id']]]+=1
    return c,len(a['images']),len(a['videos']),sha(p)

def main():
    out=RUN/'results/exp1';sources={
        'N_old':R/'work_dirs/bdd_4domain/continual_forgetting_eval/spaP1_causalOld_4domain_5e',
        'F_old':R/'work_dirs/bdd_4domain/continual_forgetting_eval/spaP1_causalOld_roiV1Weak_4domain_5e',
        'F':F/'forgetting_eval'}
    counts={d:gt_counts(d) for d in STAGES};rows=[];provenance=[]
    for model,base in sources.items():
        paths=sorted(base.rglob('summary_metrics.csv'));assert len(paths)==16,(model,len(paths))
        for p in paths:
            rel=p.relative_to(base);stage=int(re.search(r'stage(\d+)',str(rel)).group(1));domain=rel.parts[2]
            assert rel.parts[1]=='epoch_5';summary=readcsv(p);track=readcsv(p.with_name('track_metrics.csv'));hota=readcsv(p.with_name('hota_metrics.csv'))
            root=p.parents[2]
            checkpoint=(root/'checkpoint_path.txt').read_text().strip();cmd=(root/'eval_command.txt').read_text().strip()
            used_path=root/'used_config.py'
            if not used_path.exists():used_path=RUN/('code/configs/mot/qdtrack/bdd_7domain/qdtrack_faster-rcnn_r50_fpn_20e_bdd_%s.py'%domain)
            used=used_path.read_text()
            assert 'stage%02d_after_'%stage in checkpoint and checkpoint.endswith('epoch_5.pth')
            assert '--fuse-conv-bn' in cmd and 'model.tracker.init_score_thr=0.30' in cmd
            assert 'obj_score_thr=0.2' in used and 'QuasiDenseTracker' in used
            dest=out/'reused'/model/('stage%02d'%stage)/domain;dest.mkdir(parents=True,exist_ok=True)
            for name in ['summary_metrics.csv','track_metrics.csv','hota_metrics.csv']:
                shutil.copyfile(p.with_name(name),dest/name)
            for name in ['checkpoint_path.txt','eval_command.txt','used_config.py','cfg_options.txt']:
                if (root/name).exists():shutil.copyfile(root/name,dest/name)
            gc,nf,nv,ah=counts[domain]
            for cls in ['OVERALL','car']:
                ngt=sum(gc.values()) if cls=='OVERALL' else gc[cls]
                fn=int(track[cls]['FN']);rcl=float(track[cls]['Rcll'])/100
                assert abs(fn-ngt*(1-rcl))<1e-5,(model,stage,domain,cls,ngt,fn,rcl)
                idsw=int(track[cls]['IDSw'])
                row=dict(model=model,stage=stage,epoch=5,eval_domain=domain,category=cls,
                         HOTA_percent=float(summary[cls]['HOTA']),DetA_percent=float(summary[cls]['DetA']),IDF1_percent=float(summary[cls]['IDF1']),AssA_percent=float(summary[cls]['AssA']),IDSw_count=idsw,GT_count=ngt,IDSw_per_1000GT=1000*idsw/ngt,frames=nf,videos=nv,
                         detector_score_thr=0.05,tracker_init_score_thr=0.30,tracker_obj_score_thr=0.20,conv_bn_fusion=True,
                         checkpoint=checkpoint,annotation_sha256=ah,source_summary=str(p),provenance='reused_full_precision_csv',summary_sha256=sha(p))
                rows.append(row)
            provenance.append(dict(model=model,stage=stage,domain=domain,source=str(p),checkpoint=checkpoint,command=cmd,threshold_evidence='used_config.py + eval_command.txt; inherited detector test_cfg in archived evaluation code',summary_sha256=sha(p)))
    writecsv(out/'stage_domain_metrics.csv',rows)
    final=[r for r in rows if r['stage']==4 and r['eval_domain']!='night'];writecsv(out/'apr_old_domain_metrics.csv',final)
    forgetting=[]
    for m in sources:
        for cls in ['OVERALL','car']:
            for metric in ['HOTA_percent','IDF1_percent','AssA_percent']:
                rs={(r['stage'],r['eval_domain']):r[metric] for r in rows if r['model']==m and r['category']==cls}
                fs=[max(rs[i,STAGES[j-1]] for i in range(j,4))-rs[4,STAGES[j-1]] for j in range(1,4)]
                forgetting.append(dict(model=m,category=cls,metric=metric,FinalOld_percent=sum(rs[4,d] for d in STAGES[:3])/3,AvgF_percentage_points=sum(fs)/3,F_overcast_pp=fs[0],F_rainy_pp=fs[1],F_dawn_dusk_pp=fs[2],formula='mean_j=1..3[max_i=j..3 R_ij - R_4j]; no clipping',provenance='recomputed_from_full_precision_csv'))
    writecsv(out/'apr_forgetting_metrics.csv',forgetting);dump(out/'reuse_provenance.json',provenance)
    notes='''All results use stage04_after_night epoch5 for FinalOld; no features extracted, no new inference needed.\nN_old versus F_old: both global_pc_ema; batch per GPU 16 versus 14; embedded gpu_ids indicate 4 versus 6 GPUs, nominal total batches 64 versus 84; seed metadata is recorded in manifests only; not a strict APR-only training comparison.\nF: proto_shrinkage_cov_ema, num_pcs=2, alpha=.10, loss=.025, min_samples=96, cov_momentum=.98, shrinkage=.15, batch16. F is a separate reference, its delta versus N_old cannot be attributed wholly to APR.\nFrozen native student QDTrack uses QuasiDenseTracker, detector score=.05, init=.30, obj=.20, nms_conf=.35, match=.5, class-aware bisoftmax; Conv-BN fusion; 1296x720 keep ratio, no test flip.\nHistorical predictions used ordinary QDTrack evaluation config, so adapter teacher and training-only ICS/APR are not executed. This is compatible student deployment.\nGT_count computed from six-class dataset annotation filtering: skip ignore/iscrowd, no image intersection, area<=0, w/h<1. Independently verified FN=(1-recall)*GT from historical CLEAR metrics. OVERALL uses original official combined result, never class mean.\nTarget evaluation revisits original adaptation pools, not independent target generalization.\nOnly learned-domain cells enter forgetting statistics. Future-domain historical cells are preserved without additional inference.\nReuse is supported by checkpoint_path.txt, epoch, used_config.py, commands, evaluator source, counts and CSV hashes. Historical checkpoint hashes were not recorded by original evaluation, so exact historical binary provenance cannot be proved retroactively; no evidence of checkpoint mismatch was found.\n'''
    (out/'comparison_config_notes.txt').write_text(notes)
    print(json.dumps(forgetting,indent=2));print('COMPLETE exp1',len(paths)*3,'matrix cells',len(rows),'rows')

if __name__=='__main__':main()
