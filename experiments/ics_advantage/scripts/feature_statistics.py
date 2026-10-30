"""Intervention invariance, P2 cross-view retrieval and exact identity silhouette."""
import argparse,json,hashlib
from common import *
from statistics_utils import weights,interval,ratio_replicates,cosine_silhouette

VIEWS=['fap','photo']+['dark_'+s for s in ['0p80','0p60','0p40','0p25']]+['fog_'+s for s in ['0p15','0p30','0p45','0p60']]+['fapk_'+s for s in ['0p90','0p70','0p50','0p30']]
META=['video','instance_id','frame_id','image_id','annotation_id','bbox','category']

def load(model):
    pieces=[]
    for domain in DOMAINS:
        with np.load(str(ROOT/'features'/model/(domain+'.npz')),allow_pickle=False) as z:
            d={k:z[k] for k in META+['feat_clean']+['feat_'+v for v in VIEWS]}
        d['weather']=np.repeat(domain,len(d['video']));pieces.append(d)
    return {k:np.concatenate([d[k] for d in pieces]) for k in pieces[0]}

def validate(data,reference,model):
    for k in META+['weather']:assert np.array_equal(data[k],reference[k]),(model,k)
    assert np.all(data['category']==1)
    for v in VIEWS:
        assert np.array_equal(np.isfinite(data['feat_'+v]).all(1),np.isfinite(reference['feat_'+v]).all(1)),(model,v,'dose support changed')
    assert np.isfinite(data['feat_clean']).all()
    for domain in DOMAINS:
        ann=json.loads((ROOT/'inputs'/('probe_'+domain+'.json')).read_text())
        mask=data['weather']==domain
        assert set(data['annotation_id'][mask])=={a['id'] for a in ann['annotations']},(model,domain,'not every GT car retained')
    split=json.loads((ROOT/'manifests/probe_samples.json').read_text())
    allowed={d['domain']:set(d['test_videos']) for d in split}
    for weather,video in zip(data['weather'],data['video']):
        assert video in allowed[weather],('Unexpected fit-video row',weather,video)

def nearest_other(clean,data):
    groups=np.array(['%s|%s|%d'%(w,v,f) for w,v,f in zip(data['weather'],data['video'],data['frame_id'])])
    nearest=np.full(len(clean),np.nan)
    for group in np.unique(groups):
        ix=np.where(groups==group)[0];ids=data['instance_id'][ix]
        distance=np.sqrt(np.maximum(0,2-2*(clean[ix]@clean[ix].T)))
        distance[ids[:,None]==ids[None,:]]=np.inf
        values=distance.min(axis=1);values[~np.isfinite(values)]=np.nan;nearest[ix]=values
    return nearest

def bootstrap_row(model,pool,view,num,den,aps,nq,baseaps,w):
    ir=float(num.sum()/den.sum());mapv=100*float(aps.sum()/nq.sum());baseline=100*float(baseaps.sum()/nq.sum())
    irboot=ratio_replicates(num,den,w);mapboot=100*ratio_replicates(aps,nq,w)
    il,ih,ni=interval(irboot);ml,mh,nm=interval(mapboot)
    row=dict(model=model,pool=pool,view=view,IR=ir,IR_ci_low=il,IR_ci_high=ih,IR_bootstrap_valid=ni,
             cross_mAP=mapv,cross_mAP_ci_low=ml,cross_mAP_ci_high=mh,mAP_bootstrap_valid=nm,
             clean_support_mAP=baseline,delta_from_clean_mAP=mapv-baseline,queries=int(nq.sum()),query_videos=int((nq>0).sum()),IR_videos=int((den>0).sum()))
    return row,irboot,mapboot

def main(batch):
    models=MODELS if batch=='1' else [m for m in ['HF_stage4','INT_stage4','HF_stage1','INT_stage1']
               if m in json.loads((ROOT/'manifests/batch2_model_provenance.json').read_text())]
    root=ROOT/'analysis'/('batch'+batch)/'features'
    reference=load(models[0]);validate(reference,reference,models[0])
    results=[];byvideo=[];silhouettes=[];means=[];effects=[];allboot={};support_protocol={};selection_cache={}
    for model in models:
        print('FEATURE_MODEL',model,flush=True)
        data=load(model);validate(data,reference,model)
        clean=probe.normalize(data['feat_clean'].astype(float));nearest=nearest_other(clean,data)
        for pool in ['night','dawn_dusk','pooled']:
            ix=np.where(np.ones(len(clean),bool) if pool=='pooled' else data['weather']==pool)[0]
            vids=np.unique(data['video'][ix]);assert len(vids)==len(np.unique(np.char.add(data['weather'][ix],np.char.add('|',data['video'][ix]))))
            vidmap={v:i for i,v in enumerate(vids)};local_vid=np.array([vidmap[v] for v in data['video'][ix]])
            w=weights(len(vids));view_irboots=[];view_mapboots=[];view_ir=[];view_map=[]
            for view in ['clean']+VIEWS:
                key='feat_'+view;available=np.isfinite(data[key][ix]).all(1);subset=ix[available]
                signature=hashlib.sha256(available.tobytes()).hexdigest();cache_key=(pool,signature)
                if cache_key not in selection_cache:
                    tasks=probe.build_retrieval(data['video'][subset],data['frame_id'][subset],data['instance_id'][subset],4,30,4000,0)
                    assert tasks,(pool,view,'no eligible P2 queries')
                    selection_cache[cache_key]=tasks
                    support_protocol[pool+'_'+signature[:12]]=dict(pool=pool,views=[],available_rois=len(subset),queries=len(tasks),
                         query_annotation_ids=[int(data['annotation_id'][subset[q]]) for q,g,pos in tasks],
                         query_videos=sorted(set(str(data['video'][subset[q]]) for q,g,pos in tasks)),video_order=vids.tolist())
                entry=support_protocol[pool+'_'+signature[:12]]
                if view not in entry['views']:entry['views'].append(view)
                tasks=selection_cache[cache_key];alter=probe.normalize(data[key][subset].astype(float));qclean=clean[subset]
                aps=np.zeros(len(vids));baseaps=np.zeros(len(vids));nq=np.zeros(len(vids))
                for q,g,pos in tasks:
                    j=vidmap[data['video'][subset[q]]]
                    aps[j]+=probe.average_precision(alter[g]@qclean[q],pos)
                    baseaps[j]+=probe.average_precision(qclean[g]@qclean[q],pos);nq[j]+=1
                if view=='clean':
                    lo,hi,nb=interval(100*ratio_replicates(aps,nq,w))
                    results.append(dict(model=model,pool=pool,view='clean',IR=0.,IR_ci_low=0.,IR_ci_high=0.,IR_bootstrap_valid=NBOOT,
                         cross_mAP=100*float(aps.sum()/nq.sum()),cross_mAP_ci_low=lo,cross_mAP_ci_high=hi,mAP_bootstrap_valid=nb,
                         clean_support_mAP=100*float(aps.sum()/nq.sum()),delta_from_clean_mAP=0.,queries=int(nq.sum()),query_videos=int((nq>0).sum()),IR_videos=0))
                    continue
                eligible=np.isfinite(nearest[subset]);valid=subset[eligible];delta=np.linalg.norm(alter[eligible]-clean[valid],axis=1)
                vi=np.array([vidmap[v] for v in data['video'][valid]])
                num=np.bincount(vi,weights=delta,minlength=len(vids));den=np.bincount(vi,weights=nearest[valid],minlength=len(vids))
                nroi=np.bincount(vi,minlength=len(vids))
                row,ib,mb=bootstrap_row(model,pool,view,num,den,aps,nq,baseaps,w)
                row.update(valid_IR_rois=len(valid),available_rois=len(subset),test_videos=len(vids),
                     mean_intervention_distance=float(num.sum()/len(valid)),mean_nearest_other_clean_distance=float(den.sum()/len(valid)));results.append(row)
                for j,video in enumerate(vids):byvideo.append(dict(model=model,pool=pool,view=view,video=str(video),valid_IR_rois=int(nroi[j]),
                    intervention_distance_sum=float(num[j]),nearest_other_distance_sum=float(den[j]),AP_sum=float(aps[j]),queries=int(nq[j]),clean_AP_sum=float(baseaps[j])))
                view_irboots.append(ib);view_mapboots.append(mb);view_ir.append(row['IR']);view_map.append(row['cross_mAP'])
                allboot[model,pool,view,'IR']=(row['IR'],ib);allboot[model,pool,view,'mAP']=(row['cross_mAP'],mb)
            mixed=[];labels=[]
            names=np.array(['%s|%d'%(v,i) for v,i in zip(data['video'][ix],data['instance_id'][ix])])
            for view in ['clean']+VIEWS:
                keep=np.isfinite(data['feat_'+view][ix]).all(1)
                mixed.append(probe.normalize(data['feat_'+view][ix][keep].astype(float)));labels.append(names[keep])
            X=np.concatenate(mixed);ll=np.concatenate(labels)
            score=float(cosine_silhouette(X,ll).mean())
            silhouettes.append(dict(model=model,pool=pool,silhouette=score,samples=len(X),identities=len(np.unique(ll)),
                                     metric='cosine',exact=True,views=15))
            ib=np.mean(view_irboots,axis=0);mb=np.mean(view_mapboots,axis=0)
            il,ih,_=interval(ib);ml,mh,_=interval(mb)
            means.append(dict(model=model,pool=pool,mean_IR=float(np.mean(view_ir)),IR_ci_low=il,IR_ci_high=ih,
                 mean_cross_mAP=float(np.mean(view_map)),mAP_ci_low=ml,mAP_ci_high=mh,silhouette=score,views=14))
            allboot[model,pool,'mean','IR']=(float(np.mean(view_ir)),ib);allboot[model,pool,'mean','mAP']=(float(np.mean(view_map)),mb)
            print('FEATURE_POOL_DONE',model,pool,score,flush=True)
    pairs=[(m,c) for m in models for c in ['C0','C4'] if m!=c] if batch=='1' else [(a,b) for a,b in [('INT_stage4','HF_stage4'),('INT_stage1','HF_stage1')] if a in models and b in models]
    for model,control in pairs:
        for pool in ['night','dawn_dusk','pooled']:
            for view in VIEWS+['mean']:
                for metric in ['IR','mAP']:
                    a,ab=allboot[model,pool,view,metric];b,bb=allboot[control,pool,view,metric]
                    lo,hi,reps=interval(ab-bb)
                    effects.append(dict(model=model,reference=control,pool=pool,view=view,metric=metric,delta=a-b,
                         ci_low=lo,ci_high=hi,bootstrap_valid_replicates=reps,CI_excludes_zero_in_favorable_direction=hi<0 if metric=='IR' else lo>0))
    # Clean baseline has no IR support fields; make every output row rectangular.
    keys=list(dict.fromkeys(k for r in results for k in r))
    csvout(root/'per_view.csv',[{k:r.get(k,'') for k in keys} for r in results])
    for model in models:csvout(root/'per_video'/(model+'.csv'),[r for r in byvideo if r['model']==model])
    csvout(root/'silhouette.csv',silhouettes)
    csvout(root/'model_means.csv',means);csvout(root/'paired_effects.csv',effects)
    dump(root/'query_supports.json',support_protocol)
    dump(root/'complete.json',dict(status='complete',models=models,bootstrap=NBOOT,seed=0,exact_cosine_silhouette=True,
           all_models_metadata_exact=True,full_test_GT_boxes=True,view_average='equal mean of14 intervention views',
           note='Dose retrieval uses only first-five sampled frame galleries; a separate matched-support clean baseline is reported. t-SNE supplemental dose features are excluded.'))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--batch',choices=['1','2'],default='1');main(p.parse_args().batch)
