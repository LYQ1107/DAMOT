"""Fixed-dose curves, summary bars and preselected intervention t-SNE panels."""
import argparse,json
from common import *
from feature_statistics import load
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
plt.rcParams.update({'font.size':10,'pdf.fonttype':42})
COLORS={'C0':'#1f77b4','C1':'#ff7f0e','C2':'#2ca02c','C3':'#d62728','C4':'#9467bd',
        'HF_stage4':'#1f77b4','INT_stage4':'#2ca02c','HF_stage1':'#1f77b4','INT_stage1':'#2ca02c'}

def save(fig,root,name):
    root.mkdir(parents=True,exist_ok=True)
    fig.savefig(str(root/(name+'.pdf')),bbox_inches='tight')
    fig.savefig(str(root/(name+'.png')),dpi=200,bbox_inches='tight');plt.close(fig)

def main(batch,skip_tsne=False):
    root=ROOT/'analysis'/('batch'+batch)/'features';out=ROOT/'figures'/('batch'+batch)
    rr=readcsv(root/'per_view.csv');means=readcsv(root/'model_means.csv')
    available={r['model'] for r in means}
    groups=[['C0','C4','C2','C1','C3']] if batch=='1' else [g for g in [['HF_stage4','INT_stage4'],['HF_stage1','INT_stage1']] if set(g)<=available]
    for models in groups:
        suffix='' if batch=='1' else '_'+models[0].split('_')[-1]
        for pool in ['night','dawn_dusk','pooled']:
            fig,axes=plt.subplots(1,3,figsize=(12,3.7),sharey=True)
            for ax,family in zip(axes,['dark','fog','fapk']):
                for model in models:
                    data=[r for r in rr if r['model']==model and r['pool']==pool and r['view'].startswith(family+'_')]
                    def strength(r):
                        raw=float(r['view'].split('_')[1].replace('p','.'))
                        return raw if family=='fog' else 1-raw
                    data.sort(key=strength)
                    x=[0.]+[strength(r) for r in data];y=[0.]+[float(r['IR']) for r in data]
                    lo=[0.]+[float(r['IR_ci_low']) for r in data];hi=[0.]+[float(r['IR_ci_high']) for r in data]
                    style='--' if model in ['C1','C3'] else '-'
                    ax.plot(x,y,style+'o',ms=4,color=COLORS[model],label=model)
                    ax.fill_between(x,lo,hi,color=COLORS[model],alpha=.12)
                ax.set_title(family);ax.set_xlabel('Strength'+(' (fog blend)' if family=='fog' else ' (1 − multiplier)' if family=='dark' else ' (1 − keep probability)'));ax.grid(alpha=.2)
            axes[0].set_ylabel('Intervention invariance ratio (IR)')
            axes[-1].legend(fontsize=8)
            fig.suptitle(pool+': pointwise video-bootstrap 95% CI; lower IR is better; one training seed')
            fig.tight_layout();save(fig,out,'dose_response_'+pool+suffix)
            fig,axes=plt.subplots(1,3,figsize=(12,3.7))
            for ax,metric,label,lo_key,hi_key in zip(axes,['mean_IR','mean_cross_mAP','silhouette'],
                    ['Mean IR (↓)','Mean cross-intervention mAP (%) (↑)','Identity cosine silhouette (↑)'],
                    ['IR_ci_low','mAP_ci_low',None],['IR_ci_high','mAP_ci_high',None]):
                data=[next(r for r in means if r['model']==model and r['pool']==pool) for model in models]
                y=np.array([float(r[metric]) for r in data])
                error=np.array([y-np.array([float(r[lo_key]) for r in data]),np.array([float(r[hi_key]) for r in data])-y]) if lo_key else None
                ax.bar(np.arange(len(models)),y,yerr=error,capsize=3,color=[COLORS[m] for m in models])
                ax.set_xticks(np.arange(len(models)));ax.set_xticklabels(models,rotation=25);ax.set_ylabel(label);ax.grid(axis='y',alpha=.2)
                if metric=='silhouette':ax.axhline(0,color='k',lw=.7)
            fig.suptitle(pool+': equal mean over 14 intervention views; IR/mAP error bars are video-bootstrap 95% CI')
            fig.tight_layout();save(fig,out,'invariance_bars_'+pool+suffix)
        if not skip_tsne:tsne(models[:3] if batch=='1' else models,root,out,'intervention_tsne'+suffix)

def tsne(models,root,out,name):
    from sklearn.decomposition import PCA
    from sklearn.manifold import TSNE
    selected=json.loads((ROOT/'manifests/tsne_selection.json').read_text())['selected']
    background=json.loads((ROOT/'manifests/tsne_background.json').read_text())['annotation_ids']
    means=readcsv(root/'model_means.csv');all_rows=[];protocol=[]
    markers=['o','s','^','D'];views=['clean','dark_0p40','fog_0p45','fapk_0p50'];display=['clean','dark 0.40','fog 0.45','FAP keep 0.50']
    fig,axes=plt.subplots(1,len(models),figsize=(4.8*len(models),4.9),squeeze=False)
    reference=None
    for model,ax in zip(models,axes[0]):
        with np.load(str(ROOT/'tsne_features'/model/'night.npz'),allow_pickle=False) as z:
            extra={k:z[k] for k in ['annotation_id']+['feat_'+v for v in views]}
        assert set(extra['annotation_id'])=={r['annotation_id'] for r in selected}
        lookup={int(a):i for i,a in enumerate(extra['annotation_id'])}
        d=load(model);clean=probe.normalize(d['feat_clean'].astype(float))
        mapping={int(a):i for i,a in enumerate(d['annotation_id']) if d['weather'][i]=='night'}
        points=[]
        for r in selected:
            for view in views:points.append(extra['feat_'+view][lookup[r['annotation_id']]])
        points=probe.normalize(np.array(points,dtype=float));base=clean[[mapping[i] for i in background]]
        X=np.concatenate([base,points]);reduced=PCA(n_components=50,random_state=0).fit_transform(X)
        estimator=TSNE(n_components=2,perplexity=30,init='pca',random_state=0,n_jobs=2)
        T=estimator.fit_transform(reduced);n=len(base)
        ax.scatter(T[:n,0],T[:n,1],s=5,color='.75',alpha=.35,rasterized=True)
        for j,r in enumerate(selected):
            pos=T[n+j*4:n+(j+1)*4];color=plt.get_cmap('tab10')(r['rank'])
            for k in range(1,4):ax.plot(pos[[0,k],0],pos[[0,k],1],color=color,lw=.5,alpha=.5)
            for k,marker in enumerate(markers):ax.scatter(pos[k,0],pos[k,1],marker=marker,s=24,color=color,edgecolors='white',linewidths=.25,zorder=3)
            for k,view in enumerate(views):all_rows.append(dict(model=model,annotation_id=r['annotation_id'],video=r['video'],instance_id=r['instance_id'],track_rank=r['rank'],view=view,x=float(pos[k,0]),y=float(pos[k,1]),background=False))
        for j,a in enumerate(background):all_rows.append(dict(model=model,annotation_id=a,video='',instance_id='',track_rank='',view='clean',x=float(T[j,0]),y=float(T[j,1]),background=True))
        row=next(r for r in means if r['model']==model and r['pool']=='night')
        ax.set_title('%s\nNight mean IR %.3f; cross mAP %.2f%%'%(model,float(row['mean_IR']),float(row['mean_cross_mAP'])),fontsize=10)
        ax.set_xticks([]);ax.set_yticks([])
        protocol.append(dict(model=model,points=len(X),highlighted_rois=40,background=len(base),PCA_dimensions=50,
              TSNE_params=estimator.get_params(),KL_divergence=float(estimator.kl_divergence_),
              title_metrics='Night population, equal mean over14 requested intervention views; not t-SNE point distances'))
    handles=[Line2D([0],[0],marker=m,color='.4',ls='',label=v,ms=6) for m,v in zip(markers,display)]
    fig.legend(handles=handles,loc='lower center',ncol=4,frameon=False,bbox_to_anchor=(.5,0))
    fig.suptitle('Fixed ten longest Night test tracks; independent t-SNE per model; qualitative only')
    fig.tight_layout(rect=(0,.06,1,.94));save(fig,out,name)
    csvout(root/(name+'_coordinates.csv'),all_rows);dump(root/(name+'_protocol.json'),protocol)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--batch',choices=['1','2'],default='1');p.add_argument('--skip-tsne',action='store_true');a=p.parse_args();main(a.batch,a.skip_tsne)
