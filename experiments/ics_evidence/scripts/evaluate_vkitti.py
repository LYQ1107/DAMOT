"""IoU-paired counterfactuals and cross-weather retrieval; no cross-weather ID assumption."""
import argparse
import json
from pathlib import Path
import numpy as np
import ics_probe_analyze as p
from evidence_cpu import ROOT, bases, csvout, dump, sign_basis
from report import rows

DOMAINS=['clone','fog','morning','overcast','rain','sunset']


def load(label,views=False):
    out={}
    for domain in DOMAINS:
        with np.load(str(ROOT/'features'/label/(domain+'.npz')),allow_pickle=False) as z:
            keys=['video','scene','frame_id','image_id','annotation_id','instance_id','bbox','category']
            out[domain]={k:z[k] for k in keys}
            out[domain]['Z']=p.normalize(z['feat_clean'].astype(float))
            if views:
                out[domain]['fap']=p.normalize(z['feat_fap'].astype(float))
                out[domain]['photo']=p.normalize(z['feat_photo'].astype(float))
    for d in out.values():
        assert np.all(d['category']==1)
        assert len(np.unique(d['annotation_id']))==len(d['annotation_id'])
    return out


def bdd_bases(label):
    if label in ['F','HF']:
        return {name:[np.load(str(ROOT/'part1'/label/'bases'/('%s_%02d.npy'%(name,i))))
                      for i in range(10 if name=='R' else 1)] for name in ['INT','ICS','W','ID','R']}
    data,bank=p.load_label(str(ROOT/'features'),'VK_self_BDD',
                           ['clear_holdout_45v','overcast','rainy','dawn_dusk','night'],[1])
    Z=p.normalize(data['feat_clean'].astype(float));fit=np.ones(len(Z),dtype=bool)
    import hashlib
    assert all(int(hashlib.md5(('%s/0'%v).encode()).hexdigest()[:8],16)%2==0 for v in np.unique(data['video']))
    bb,stats,_=bases(data,bank,Z,fit)
    root=ROOT/'part2'/label/'BDD_bases';root.mkdir(parents=True,exist_ok=True)
    for name,values in bb.items():
        for i,E in enumerate(values):np.save(str(root/('%s_%02d.npy'%(name,i))),E)
    dump(root/'protocol.json',dict(fit_videos=np.unique(data['video'][fit]).tolist(),stats=stats,
          meaning='Native VKITTI final checkpoint also evaluated on BDD to estimate the same BDD comparator families'))
    return bb


def part2():
    all_rows,replicates,gates=[],[],[]
    raw_pairs=rows(ROOT/'part2/GT_pairs.csv')
    labels=['F','HF']+(['VK_self'] if (ROOT/'features/VK_self/meta.json').exists() else [])
    protocols=[]
    for label in labels:
        ds=load(label,True);clone=ds['clone'];B=bdd_bases(label)
        # Entire physical scenes are kept together across weather and both cameras.
        fit=p.split_mask(clone['scene'],0);te=~fit
        residual=np.concatenate([clone[v][fit]-clone['Z'][fit] for v in ['fap','photo']])
        B['INT-vk']=[sign_basis(p.top_eig(residual.T@residual/len(residual),2))]
        root=ROOT/'part2'/label;root.mkdir(parents=True,exist_ok=True)
        np.save(str(root/'INT_vk.npy'),B['INT-vk'][0])
        image_key=np.array(['%s|%d'%(v,i) for v,i in zip(clone['video'][te],clone['image_id'][te])])
        diff=p.identity_pairs(clone['Z'][te],image_key,clone['instance_id'][te],seed=0)
        identity_mean_sq=float(np.mean(np.sum(diff*diff,axis=1)))
        lookup={d:{int(a):i for i,a in enumerate(dd['annotation_id'])} for d,dd in ds.items()}
        test_scenes=set(clone['scene'][te]);model_rows=[]
        for domain in DOMAINS[1:]:
            selected=[r for r in raw_pairs if r['weather']==domain and r['scene'] in test_scenes]
            ci=np.array([lookup['clone'][int(r['clone_annotation_id'])] for r in selected])
            vi=np.array([lookup[domain][int(r['variant_annotation_id'])] for r in selected])
            assert np.array_equal(clone['scene'][ci],ds[domain]['scene'][vi])
            assert np.array_equal(clone['frame_id'][ci],ds[domain]['frame_id'][vi])
            delta=ds[domain]['Z'][vi]-clone['Z'][ci]
            total=float(np.sum(delta*delta));mean_sq=total/len(delta)
            for name,bb in B.items():
                values=[float(np.sum((delta@E)**2)/total) for E in bb]
                row=dict(model=label,basis=name,weather=domain,CCR=float(np.mean(values)),
                    random_replicate_std=float(np.std(values)),variant_std=None,pairs=len(delta),delta_total_sq=total,
                    delta_mean_sq=mean_sq,identity_mean_sq=identity_mean_sq,
                    weather_identity_energy_ratio=mean_sq/identity_mean_sq)
                all_rows.append(row);model_rows.append(row)
                for i,value in enumerate(values):replicates.append(dict(model=label,basis=name,weather=domain,replicate=i,CCR=value))
        means={}
        for name in B:
            group=[r for r in model_rows if r['basis']==name]
            means[name]=float(np.mean([r['CCR'] for r in group]))
            rep_means=[np.mean([r['CCR'] for r in replicates if r['model']==label and r['basis']==name and r['replicate']==i]) for i in range(len(B[name]))]
            all_rows.append(dict(model=label,basis=name,weather='mean',CCR=means[name],
                random_replicate_std=float(np.std(rep_means)),variant_std=float(np.std([r['CCR'] for r in group])),pairs=sum(r['pairs'] for r in group),
                delta_total_sq=sum(r['delta_total_sq'] for r in group),delta_mean_sq=float(np.mean([r['delta_mean_sq'] for r in group])),
                identity_mean_sq=identity_mean_sq,weather_identity_energy_ratio=float(np.mean([r['weather_identity_energy_ratio'] for r in group]))))
        passed=means['INT']>=5*means['R'] and means['INT']>means['ICS']
        gates.append(dict(model=label,INT_CCR=means['INT'],R_CCR=means['R'],ICS_CCR=means['ICS'],
                          INT_over_random=means['INT']/means['R'],status='PASS' if passed else 'FAIL'))
        protocols.append(dict(model=label,fit_scenes=sorted(set(clone['scene'][fit])),test_scenes=sorted(test_scenes),
             comparator_source='same-model BDD fit-video features; INT-vk uses clone fit scenes only',
             camera_policy='GT pairing requires identical scene, camera and frame; metric split holds out entire scenes',
             identity_reference_pairs=len(diff),identity_reference='one seed0 random different-car pair per clone test frame',
             pair_counts={d:len([r for r in raw_pairs if r['weather']==d and r['scene'] in test_scenes]) for d in DOMAINS[1:]}))
    csvout(ROOT/'part2/CCR.csv',all_rows);csvout(ROOT/'part2/CCR_replicates.csv',replicates);csvout(ROOT/'part2/gates.csv',gates)
    by_weather=[]
    for label in labels:
        for domain in DOMAINS[1:]:
            a={r['basis']:r['CCR'] for r in all_rows if r['model']==label and r['weather']==domain}
            by_weather.append(dict(model=label,weather=domain,INT_CCR=a['INT'],ICS_CCR=a['ICS'],R_CCR=a['R'],
                                   INT_over_random=a['INT']/a['R'],pass_both=a['INT']>=5*a['R'] and a['INT']>a['ICS']))
    csvout(ROOT/'part2/gates_by_weather.csv',by_weather)
    dump(ROOT/'part2/protocol.json',protocols)
    draw_A(all_rows,labels)
    primary_only='VK_self' not in labels and (ROOT/'weights/VK_self_stage5_e5.pth').exists()
    dump(ROOT/'part2'/('primary_complete.json' if primary_only else 'complete.json'),
         dict(status='primary_complete' if primary_only else 'complete',models=labels,
              pending=['VK_self'] if primary_only else []))


def draw_A(all_rows,labels):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams.update({'pdf.fonttype':42,'font.size':10})
    fig,axes=plt.subplots(1,len(labels),figsize=(4.7*len(labels),3.8),sharey=True,squeeze=False)
    names=['INT','ICS','W','ID','R','INT-vk']
    for ax,label in zip(axes[0],labels):
        mean=[];sd=[]
        for name in names:
            values=[float(r['CCR']) for r in all_rows if r['model']==label and r['basis']==name and r['weather']!='mean']
            mean.append(np.mean(values));sd.append(np.std(values))
        ax.bar(np.arange(6),mean,yerr=sd,capsize=3,color=['#1976b8','#d07821','#3c9d57','#bd4a4a','#888888','#8662aa'])
        ax.axhline(2/256,ls='--',color='k',lw=1,label='2/256')
        ax.set_xticks(np.arange(6));ax.set_xticklabels(names,rotation=25);ax.set_title(label);ax.grid(axis='y',alpha=.2)
    axes[0,0].set_ylabel('Counterfactual capture rate')
    axes[0,0].legend(loc='upper right',fontsize=8,frameon=False)
    fig.suptitle('Five weather variants: mean ± variant SD')
    fig.tight_layout()
    out=ROOT/'part2/figures';out.mkdir(exist_ok=True)
    fig.savefig(str(out/'A_counterfactual_CCR.pdf'),bbox_inches='tight')
    fig.savefig(str(out/'A_counterfactual_CCR.png'),dpi=200,bbox_inches='tight');plt.close(fig)


def part3():
    pair_rows=rows(ROOT/'part2/GT_pairs.csv');all_rows=[];means={};reference=None
    for label in ['C0','C1','C2','C3','C4','X3-1']:
        ds=load(label);clone=ds['clone'];test=~p.split_mask(clone['scene'],0)
        keys=['video','scene','frame_id','annotation_id','instance_id','bbox','category']
        if reference is None:reference=ds
        else:
            for domain in DOMAINS:
                for key in keys:assert np.array_equal(ds[domain][key],reference[domain][key]),(label,domain,key)
        same_aps=[]
        for q in np.where(test)[0]:
            delta=np.abs(clone['frame_id']-clone['frame_id'][q])
            gallery=np.where(test&(clone['scene']==clone['scene'][q])&(delta>=4)&(delta<=30))[0]
            pos=clone['instance_id'][gallery]==clone['instance_id'][q]
            if pos.any() and (~pos).any():same_aps.append(p.average_precision(clone['Z'][gallery]@clone['Z'][q],pos))
        same=100*float(np.mean(same_aps))
        lookup={d:{int(a):i for i,a in enumerate(dd['annotation_id'])} for d,dd in ds.items()}
        cross=[]
        for domain in DOMAINS[1:]:
            aps=[]
            for r in pair_rows:
                if r['weather']!=domain:continue
                ci=lookup['clone'][int(r['clone_annotation_id'])]
                if not test[ci]:continue
                q=lookup[domain][int(r['variant_annotation_id'])]
                gallery=np.where(test&(clone['scene']==r['scene'])&(np.abs(clone['frame_id']-int(r['frame_id']))<=10))[0]
                # Cross-weather identity originates in the IoU matched clone annotation.
                pos=clone['instance_id'][gallery]==clone['instance_id'][ci]
                assert pos.any() and (~pos).any()
                aps.append(p.average_precision(clone['Z'][gallery]@ds[domain]['Z'][q],pos))
            value=100*float(np.mean(aps));cross.append(value)
            all_rows.append(dict(model=label,weather=domain,cross_mAP_pp=value,clone_mAP_pp=same,queries=len(aps),clone_queries=len(same_aps)))
        means[label]=dict(cross=float(np.mean(cross)),same=same)
        all_rows.append(dict(model=label,weather='mean',cross_mAP_pp=means[label]['cross'],clone_mAP_pp=same,
                             queries=sum(r['queries'] for r in all_rows if r['model']==label),clone_queries=len(same_aps)))
    contrasts=[]
    for label in ['C1','C2','C3','C4','X3-1']:
        dc=means[label]['cross']-means['C0']['cross'];ds=means[label]['same']-means['C0']['same']
        dr=means[label]['cross']-means['C4']['cross']
        passed=dc>=1 and dr>0 and dc>ds
        contrasts.append(dict(model=label,cross_gain_vs_C0_pp=dc,clone_gain_vs_C0_pp=ds,
                              cross_gain_vs_C4_pp=dr,pass_all_conditions=passed))
    csvout(ROOT/'part3/retrieval.csv',all_rows);csvout(ROOT/'part3/contrasts.csv',contrasts)
    dump(ROOT/'part3/gate.json',dict(status='PASS' if any(r['pass_all_conditions'] and r['model'] in ['C2','C3'] for r in contrasts) else 'FAIL',
         thresholds=dict(cross_gain_vs_C0_pp=1.,must_exceed_C4=True,cross_gain_must_exceed_clone_gain=True)))
    dump(ROOT/'part3/protocol.json',dict(seed=0,split='held-out physical scenes, same as part2',
          cross_gallery='clone, same physical scene (both cameras included), abs frame difference <=10',
          clone_gallery='clone, same physical scene (both cameras included), abs frame difference 4 through30 inclusive',
          query_limit='all eligible test-scene IoU pairs, no query cap',
          positive_identity='IoU-matched clone GT instance, then propagated within the clone tracking annotation',
          no_cross_weather_instance_id_assumption=True,metadata_exact_across_all_models=True,
          X3_1_role='APR-on no-ICS reference, C0-C4 APR-off continuation'))
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams.update({'pdf.fonttype':42,'font.size':10})
    fig,ax=plt.subplots(figsize=(10,4));models=['C0','C1','C2','C3','C4','X3-1'];width=.13
    for i,model in enumerate(models):
        vals=[r['cross_mAP_pp'] for r in all_rows if r['model']==model and r['weather']!='mean']
        ax.bar(np.arange(5)+(i-2.5)*width,vals,width,label=model)
    ax.set_xticks(np.arange(5));ax.set_xticklabels(DOMAINS[1:]);ax.set_ylabel('Cross-weather identity mAP (%)')
    ax.legend(ncol=6,loc='upper center',bbox_to_anchor=(.5,1.15));ax.grid(axis='y',alpha=.2);fig.tight_layout()
    out=ROOT/'part3/figures';out.mkdir(exist_ok=True)
    fig.savefig(str(out/'E_cross_weather_retrieval.pdf'),bbox_inches='tight')
    fig.savefig(str(out/'E_cross_weather_retrieval.png'),dpi=200,bbox_inches='tight');plt.close(fig)
    dump(ROOT/'part3/complete.json',dict(status='complete',models=models))


if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--part',choices=['2','3','A'],required=True)
    args=ap.parse_args()
    if args.part=='A':
        data=rows(ROOT/'part2/CCR.csv')
        draw_A(data,list(dict.fromkeys(r['model'] for r in data)))
    else:(part2 if args.part=='2' else part3)()
