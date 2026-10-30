"""Test-video paired effects, fixed brightness thirds and video bootstrap."""
import argparse,json
from common import *
from statistics_utils import paired_summary,correlation

def cases(batch):
    if batch=='1':return [('batch1',[('C2','C0'),('C2','C4')],DOMAINS)]
    result=[]
    if 'INT_stage4' in json.loads((ROOT/'manifests/batch2_model_provenance.json').read_text()):
        result.append(('batch2_stage4',[('INT_stage4','HF_stage4')],['overcast','rainy','dawn_dusk','night']))
    result.append(('batch2_stage1',[('INT_stage1','HF_stage1')],['overcast']))
    return result

def main(batch):
    bright={(r['domain'],r['video']):float(r['brightness']) for r in readcsv(ROOT/'tracking/video_brightness.csv')}
    effects=[];details=[];corrs=[];bins=[]
    for case,pairs,domains in cases(batch):
        for new,ref in pairs:
            combined=[]
            for domain in domains:
                aa={ (r['video'],r['category']):r for r in readcsv(ROOT/'tracking'/new/domain/'per_video.csv') if r['test']=='True'}
                bb={ (r['video'],r['category']):r for r in readcsv(ROOT/'tracking'/ref/domain/'per_video.csv') if r['test']=='True'}
                assert aa.keys()==bb.keys()
                for category in ['OVERALL','car']:
                    vids=sorted(v for v,c in aa if c==category)
                    rows=[]
                    for video in vids:
                        r=dict(case=case,model=new,reference=ref,domain=domain,video=video,category=category,brightness=bright[domain,video])
                        for metric in ['HOTA','AssA','IDF1','IDSW']:
                            r[metric+'_delta']=float(aa[video,category][metric])-float(bb[video,category][metric])
                        rows.append(r);details.append(r);combined.append(r)
                    summarize(rows,effects,corrs,'all_test_videos',domain)
                    if domain=='night':
                        stratify(rows,effects,bins,'night_within_domain')
            if batch=='2' and len(domains)==4:
                for category in ['OVERALL','car']:
                    pool=[r for r in combined if r['category']==category]
                    summarize(pool,effects,corrs,'all_test_videos','all_four_domains')
                    stratify(pool,effects,bins,'four_domains_pooled')
    out=ROOT/'analysis'/('batch'+batch)/'tracking'
    csvout(out/'paired_effects.csv',effects);csvout(out/'video_differences.csv',details)
    csvout(out/'spearman.csv',corrs);csvout(out/'brightness_bins.csv',bins)
    draw(details,effects,bins,out)
    dump(out/'complete.json',dict(status='complete',batch=batch,seed=0,bootstrap_repetitions=NBOOT,
         statistic='equal-video paired mean differences; distinct from original nonlinear aggregate differences',
         test_only=True,lower_is_better=['IDSW'],all_zero_wilcoxon_p=1,CI='pointwise percentile95%, no multiplicity correction',single_training_seed=True))

def summarize(rows,effects,corrs,stratum,domain):
    base={k:rows[0][k] for k in ['case','model','reference','category']}
    for metric in ['HOTA','AssA','IDF1','IDSW']:
        effects.append(dict(base,domain=domain,stratum=stratum,metric=metric,
                            **paired_summary([r[metric+'_delta'] for r in rows],metric=='IDSW')))
        corrs.append(dict(base,domain=domain,stratum=stratum,metric=metric,
                          **correlation([r['brightness'] for r in rows],[r[metric+'_delta'] for r in rows])))

def stratify(rows,effects,bins,scope):
    ordered=sorted(rows,key=lambda r:(r['brightness'],r['domain'],r['video']))
    for label,ix in zip(['darkest','middle','brightest'],np.array_split(np.arange(len(ordered)),3)):
        selected=[ordered[i] for i in ix]
        for r in selected:bins.append({**r,'scope':scope,'brightness_third':label})
        dummy=[];summarize(selected,effects,dummy,scope+':'+label,'night' if scope=='night_within_domain' else 'all_four_domains')

def draw(details,effects,bins,out):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams.update({'font.size':10,'pdf.fonttype':42})
    combinations=list(dict.fromkeys((r['case'],r['model'],r['reference']) for r in details))
    fig,axes=plt.subplots(1,len(combinations),figsize=(6*len(combinations),4.4),squeeze=False)
    colors={'darkest':'#315d9b','middle':'#d18b30','brightest':'#54a379'}
    for ax,(case,model,ref) in zip(axes[0],combinations):
        data=[r for r in details if r['case']==case and r['model']==model and r['reference']==ref and r['category']=='OVERALL' and r['domain']=='night']
        # Stage1 has only Overcast; show all its test videos without inventing Night bins.
        if not data:data=[r for r in details if r['case']==case and r['model']==model and r['reference']==ref and r['category']=='OVERALL']
        ax.scatter([r['brightness'] for r in data],[r['AssA_delta'] for r in data],s=24,color='.55',alpha=.8)
        for group,color in colors.items():
            selected=[r for r in bins if r['case']==case and r['model']==model and r['reference']==ref and r['category']=='OVERALL' and r['scope']=='night_within_domain' and r['brightness_third']==group]
            if not selected:continue
            stat=next(r for r in effects if r['case']==case and r['model']==model and r['reference']==ref and r['category']=='OVERALL' and r['domain']=='night' and r['stratum']=='night_within_domain:'+group and r['metric']=='AssA')
            mean=stat['mean'];ax.errorbar(np.mean([r['brightness'] for r in selected]),mean,yerr=[[mean-stat['ci_low']],[stat['ci_high']-mean]],fmt='o',capsize=4,color=color,label=group)
        ax.axhline(0,color='k',lw=.8);ax.set_title(case+': '+model+' − '+ref)
        ax.set_xlabel('Original-frame video brightness (0–255)');ax.set_ylabel('Paired AssA difference (pp)');ax.grid(alpha=.15)
        if ax.get_legend_handles_labels()[0]:ax.legend(fontsize=8)
    fig.suptitle('Held-out test videos; pointwise video-bootstrap 95% CI; one training seed')
    fig.tight_layout()
    fig.savefig(str(out/'brightness_AssA.pdf'),bbox_inches='tight');fig.savefig(str(out/'brightness_AssA.png'),dpi=200,bbox_inches='tight');plt.close(fig)
    pooled=[r for r in bins if r['scope']=='four_domains_pooled' and r['category']=='OVERALL']
    if pooled:
        cases=list(dict.fromkeys((r['case'],r['model'],r['reference']) for r in pooled))
        fig,axes=plt.subplots(1,len(cases),figsize=(6*len(cases),4.4),squeeze=False)
        for ax,(case,model,ref) in zip(axes[0],cases):
            data=[r for r in pooled if r['case']==case and r['model']==model and r['reference']==ref]
            ax.scatter([r['brightness'] for r in data],[r['AssA_delta'] for r in data],s=24,color='.55',alpha=.8)
            for group,color in colors.items():
                selected=[r for r in data if r['brightness_third']==group]
                stat=next(r for r in effects if r['case']==case and r['model']==model and r['reference']==ref and r['category']=='OVERALL' and r['domain']=='all_four_domains' and r['stratum']=='four_domains_pooled:'+group and r['metric']=='AssA')
                mean=stat['mean'];ax.errorbar(np.mean([r['brightness'] for r in selected]),mean,yerr=[[mean-stat['ci_low']],[stat['ci_high']-mean]],fmt='o',capsize=4,color=color,label=group)
            ax.axhline(0,color='k',lw=.8);ax.set_title(model+' − '+ref);ax.set_xlabel('Original-frame video brightness (0–255)');ax.set_ylabel('Paired AssA difference (pp)');ax.grid(alpha=.15);ax.legend(fontsize=8)
        fig.suptitle('All four target domains pooled: held-out videos; pointwise bootstrap95% CI')
        fig.tight_layout();fig.savefig(str(out/'brightness_AssA_four_domains.pdf'),bbox_inches='tight');fig.savefig(str(out/'brightness_AssA_four_domains.png'),dpi=200,bbox_inches='tight');plt.close(fig)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--batch',choices=['1','2'],default='1');main(p.parse_args().batch)
