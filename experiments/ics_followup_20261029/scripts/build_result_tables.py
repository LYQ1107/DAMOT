"""Aggregate all prespecified cells; never select rows by measured performance."""
import csv,json,sys
from pathlib import Path
from collections import defaultdict
from protocol import write_json
ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/'results/combined_tables';OUT.mkdir(parents=True,exist_ok=True)
HF=Path('/data1/liuyeqiang/damot_apr_complement_hf_20261028_1420');DOMAINS=['overcast','rainy','dawn_dusk','night'];METRICS=['HOTA','DetA','IDF1','AssA']
def read(p):return list(csv.DictReader(Path(p).open()))
def write(p,rows):
    assert rows,p;p=Path(p);p.parent.mkdir(parents=True,exist_ok=True)
    with p.open('w',newline='') as f:w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
def subspaces():
    rows=[]
    for name,label in [('F','F_stage4_e5'),('HF','HF_stage4_e5'),('NOGEOM','NOGEOM_stage4_e5')]+[('S5-'+g,'S5_HF_'+g) for g in 'abcd']:
        folder=ROOT/'results'/('analysis_'+label+'_v2');energy=read(folder/'csv/P1_energy.csv');e=next(r for r in energy if r['subspace']=='ICS (checkpoint)');angles=next(r for r in read(folder/'csv/P1_angles.csv') if r['subspace']=='Oracle weather (labels)');p2=read(folder/'csv/P2_dissociation.csv');proj=next(r for r in p2 if r['space']=='E only (2-d)');full=next(r for r in p2 if r['space']=='Full (256-d)')
        rows.append(dict(model=name,label=label,bank_basis='actual stored bank_pcs',weather_energy=float(e['Real weather shift']),identity_energy=float(e['Identity']),weather_identity_ratio=float(e['Weather/Identity']),angle_weather_1_deg=float(angles['angle1']),angle_weather_2_deg=float(angles['angle2']),E_weather_bacc=float(proj['weather_bacc']),E_night_clear_bacc=float(proj['night_vs_clear_bacc']),E_night_rest_bacc=float(proj['night_vs_rest_bacc']),E_id_mAP=float(proj['id_mAP']),Full_weather_bacc=float(full['weather_bacc']),Full_id_mAP=float(full['id_mAP']),seed=0,analysis_directory=str(folder)))
    assert len(rows)==7;write(OUT/'table_C_subspaces.csv',rows)
def scans():
    rows=[]
    for name in ['F','HF','NOGEOM']:
        for r in read(ROOT/'results'/('analysis_%s_stage4_e5_v2'%name)/'csv/P3_intervention.csv'):rows.append(dict(model=name,**r))
    assert len(rows)==234;write(OUT/'table_D_scan.csv',rows)
def tracking():
    rows=[];raw=[]
    for name in ['F','HF']:
        for tag in ['baseline']+['%s_beta%g'%(b,v) for b in ['ICS_stored_m2','FAP_clear_fit_m2','Identity_fit_m16','Random_m2'] for v in [.5,1.]]:
            for domain in DOMAINS:
                folder=ROOT/'results/S4'/name/tag/domain
                assert json.loads((folder/'complete.json').read_text())['status']=='complete'
                rows+=read(folder/'metrics.csv')
                for r in read(folder/'evaluation/summary_metrics.csv'):raw.append(dict(checkpoint=name,setting=tag,domain=domain,category=r[''],**{m+'_percent':r[m] for m in METRICS}))
    assert len(rows)==144 and len(raw)==504;write(OUT/'table_F_all_domains.csv',rows);write(OUT/'S4_full_class_metrics.csv',raw)
    grouped=defaultdict(list)
    for r in rows:grouped[(r['checkpoint'],r['basis'],r['beta'],r['category'])].append(r)
    avg=[]
    for (name,basis,beta,category),rr in grouped.items():
        assert len(rr)==4 and {r['domain'] for r in rr}==set(DOMAINS)
        row=dict(checkpoint=name,basis=basis,beta=beta,category=category,aggregation='equal mean of four domains (not official pooled HOTA)')
        for m in METRICS:
            row[m+'_percent']=sum(float(r[m+'_percent']) for r in rr)/4;row['delta_'+m+'_pp']=sum(float(r['delta_'+m+'_pp']) for r in rr)/4
        avg.append(row)
    write(OUT/'table_F_four_domain_means.csv',avg)
    write(OUT/'S4_F_new_vs_archived_baseline.csv',read(ROOT/'results/S4/F/fresh_vs_historical_baseline.csv'))
def short_training():
    values=[];forget=[];bank=[];allcells={}
    old={}
    for d in DOMAINS[:3]:
        for r in read(HF/'results/stage3'/d/'metrics.csv'):old[(d,r['category'])]=r
    for g in 'abcd':
        folder=ROOT/'results/S5'/('HF_'+g);done=json.loads((folder/'train_complete.json').read_text());assert done['epochs']==2
        for d in DOMAINS:
            for r in read(folder/'results/stage4'/d/'metrics.csv'):
                assert r['epoch']=='2';row=dict(group=g,**r);values.append(row);allcells[(g,d,r['category'])]=r
                if d!='night':
                    baseline=old[(d,r['category'])];assert baseline['checkpoint_sha256']==json.loads((folder/'manifests/common.json').read_text())['checkpoint_sha256']
                    f=dict(group=g,domain=d,category=r['category'],baseline_stage=3,trained_stage=4,trained_epochs=2)
                    for m in METRICS:f['stage3_'+m+'_percent']=float(baseline[m+'_percent']);f['stage4_'+m+'_percent']=float(r[m+'_percent']);f['forgetting_'+m+'_pp']=f['stage3_'+m+'_percent']-f['stage4_'+m+'_percent']
                    forget.append(f)
        for rank in range(4):
            start=json.loads((folder/('start_rank%d.json'%rank)).read_text());end=json.loads((folder/('end_rank%d.json'%rank)).read_text());assert start['all_loaded_tensors_exact'] and start['num_updates']==660
            active=end['activity']['APR_coupled_loss_active_calls']-start['activity']['APR_coupled_loss_active_calls']
            assert (active>0) if g in 'ac' else active==0
            if g in 'cd':
                pair_rows=read(folder/('pair_checks_rank%d.csv'%rank));assert len(pair_rows)==end['pair_checks'];assert sum(int(r['insufficient_pairs']) for r in pair_rows)==end['insufficient_pairs'];assert {r['min_samples'] for r in pair_rows}=={'96'}
            bank.append(dict(group=g,rank=rank,num_updates_start=end['num_updates_start'],num_updates_end=end['num_updates_end'],updates=end['updates'],pair_checks=end['pair_checks'],insufficient_pairs=end['insufficient_pairs'],APR_coupled_active_calls_in_this_run=active,history_sha256=start['history_sha256'],source_checkpoint_sha256=start['checkpoint_sha256']))
    assert len(values)==112 and len(forget)==84 and len(bank)==16
    assert len({r['history_sha256'] for r in bank})==1
    write(OUT/'table_G_all_domains.csv',values);write(OUT/'table_G_forgetting.csv',forget);write(OUT/'S5_bank_updates_and_insufficient_pairs.csv',bank)
    effects=[]
    for label,plus,minus in [('coupling_proto','a','b'),('coupling_pairwise','c','d'),('estimator_coupled','c','a'),('estimator_uncoupled','d','b')]:
        for d in DOMAINS:
            for cat in ['pedestrian','car','truck','bus','motorcycle','bicycle','OVERALL']:
                a,b=allcells[(plus,d,cat)],allcells[(minus,d,cat)];row=dict(contrast=label,formula=plus+' - '+minus,domain=d,category=cat)
                for m in METRICS:row[m+'_effect_pp']=float(a[m+'_percent'])-float(b[m+'_percent'])
                effects.append(row)
    write(OUT/'S5_factorial_effects.csv',effects)
    summary=[];sub={r['model']:r for r in read(OUT/'table_C_subspaces.csv')}
    for g in 'abcd':
        for cat in ['car','OVERALL']:
            row=dict(group=g,category=cat,estimator='proto_shrinkage_cov_ema' if g in 'ab' else 'pairwise_residual',coupling=g in 'ac',epochs=2,seed=1790355234)
            for m in METRICS:
                row['night_'+m+'_percent']=float(allcells[(g,'night',cat)][m+'_percent']);row['old_mean_'+m+'_percent']=sum(float(allcells[(g,d,cat)][m+'_percent']) for d in DOMAINS[:3])/3;row['old_mean_forgetting_'+m+'_pp']=sum(float(r['forgetting_'+m+'_pp']) for r in forget if r['group']==g and r['category']==cat)/3
            for k in ['weather_identity_ratio','angle_weather_1_deg','angle_weather_2_deg']:row[k]=sub['S5-'+g][k]
            summary.append(row)
    write(OUT/'table_G_summary.csv',summary)
def main():
    assert (ROOT/'manifests/experiments_complete.json').exists(),'Full experiment queue is not complete; no final tables will be fabricated'
    subspaces();scans();tracking();short_training();write_json(OUT/'complete.json',dict(status='complete',selection_by_metric=False,all_seven_model_subspaces=True,all_234_scan_cells=True,all_72_S4_cells=True,all_four_S5_groups=True))
    print('ALL PRESPECIFIED RESULT TABLES BUILT')
if __name__=='__main__':main()
