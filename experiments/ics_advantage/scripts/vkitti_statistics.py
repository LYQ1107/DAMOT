"""Second-batch fixed VKITTI counterfactual retrieval and four-weather t-SNE."""
import argparse,json
from collections import Counter,defaultdict
from common import *
PREVIOUS=Path('/data1/liuyeqiang/ics_evidence_20261030_215311')
WEATHERS=['clone','fog','morning','overcast','rain','sunset']

def freeze_selection():
    records=readcsv(PREVIOUS/'part2/GT_pairs.csv')
    scenes=np.array(sorted({r['scene'] for r in records}));test=set(scenes[~probe.split_mask(scenes,0)])
    rr=[r for r in records if r['scene'] in test]
    counts=Counter((r['scene'],int(r['clone_instance_id'])) for r in rr)
    weather_maps={d:{int(r['clone_annotation_id']):r for r in rr if r['weather']==d} for d in ['fog','rain','sunset']}
    common=set.intersection(*(set(v) for v in weather_maps.values()));byid=defaultdict(list)
    for aid in common:
        r=weather_maps['fog'][aid];byid[r['scene'],int(r['clone_instance_id'])].append(aid)
    tracks=sorted(byid,key=lambda k:(-counts[k],k))[:10];assert len(tracks)==10
    selected=[]
    for rank,key in enumerate(tracks):
        aids=sorted(byid[key],key=lambda a:(int(weather_maps['fog'][a]['frame_id']),int(weather_maps['fog'][a]['camera']),a))
        aid=aids[0];r=weather_maps['fog'][aid]
        selected.append(dict(rank=rank,scene=key[0],clone_instance_id=key[1],pair_count=counts[key],frame_id=int(r['frame_id']),camera=int(r['camera']),
                  annotations={'clone':aid,**{d:int(weather_maps[d][aid]['variant_annotation_id']) for d in weather_maps}}))
    data=json.loads((PREVIOUS/'data/vkitti2/annotations/clone.json').read_text())
    videos={v['id']:v['name'].split('_')[0] for v in data['videos']};images={i['id']:i for i in data['images']}
    selected_scenes={r['scene'] for r in selected};highlighted={r['annotations']['clone'] for r in selected}
    candidates=sorted(a['id'] for a in data['annotations'] if videos[images[a['image_id']]['video_id']] in selected_scenes and a['id'] not in highlighted)
    n=min(1000,len(candidates));background=np.random.RandomState(0).choice(candidates,n,replace=False).tolist()
    dump(ROOT/'manifests/vk_tsne_selection.json',dict(test_scenes=sorted(test),seed=0,
         rule='top10 tracks by total heldout IoU pair count across five variants, lexical identity tie-break; earliest common frame then camera for the four displayed weathers',
         selected=selected,background_annotation_ids=background,background_eligible=len(candidates),background_shortfall=1000-n,
         pairing_source=str(PREVIOUS/'part2/GT_pairs.csv'),pairing_sha256=sha(PREVIOUS/'part2/GT_pairs.csv')))
    print('VK_SELECTION_FROZEN',len(selected),n,sorted(test),flush=True)

def load(model):
    result={}
    for domain in WEATHERS:
        with np.load(str(ROOT/'vk_features'/model/(domain+'.npz')),allow_pickle=False) as z:
            result[domain]={k:z[k] for k in ['video','scene','frame_id','annotation_id','instance_id','bbox','category']}
            result[domain]['Z']=probe.normalize(z['feat_clean'].astype(float))
        assert set(result[domain]['scene'])=={'Scene01','Scene02','Scene06','Scene18','Scene20'}
    return result

def main():
    supplied=json.loads((ROOT/'manifests/batch2_model_provenance.json').read_text())
    models=[m for m in ['HF_stage4','INT_stage4','HF_stage1','INT_stage1'] if m in supplied];reference=None;output=[]
    selection=json.loads((ROOT/'manifests/vk_tsne_selection.json').read_text());pair_rows=readcsv(PREVIOUS/'part2/GT_pairs.csv')
    for model in models:
        ds=load(model);clone=ds['clone'];test=~probe.split_mask(clone['scene'],0)
        if reference is None:reference=ds
        else:
            for d in WEATHERS:
                for key in ['video','scene','frame_id','annotation_id','instance_id','bbox','category']:assert np.array_equal(ds[d][key],reference[d][key]),(model,d,key)
        assert set(clone['scene'][test])==set(selection['test_scenes'])
        look={d:{int(a):i for i,a in enumerate(r['annotation_id'])} for d,r in ds.items()}
        aps=[];missing=0
        for q in np.where(test)[0]:
            gap=np.abs(clone['frame_id']-clone['frame_id'][q]);g=np.where(test&(clone['scene']==clone['scene'][q])&(gap>=4)&(gap<=30))[0]
            pos=clone['instance_id'][g]==clone['instance_id'][q]
            if pos.any():aps.append(probe.average_precision(clone['Z'][g]@clone['Z'][q],pos))
            else:missing+=1
        assert len(aps)==1175 and missing==4
        same=100*float(np.mean(aps));values=[]
        for d in WEATHERS[1:]:
            ap=[];trivial=0
            for r in pair_rows:
                if r['weather']!=d:continue
                c=look['clone'][int(r['clone_annotation_id'])]
                if not test[c]:continue
                q=look[d][int(r['variant_annotation_id'])]
                g=np.where(test&(clone['scene']==r['scene'])&(np.abs(clone['frame_id']-int(r['frame_id']))<=10))[0]
                pos=clone['instance_id'][g]==clone['instance_id'][c];assert pos.any();trivial+=int(pos.all())
                ap.append(probe.average_precision(clone['Z'][g]@ds[d]['Z'][q],pos))
            assert len(ap)==1179 and trivial==10
            value=100*float(np.mean(ap));values.append(value)
            output.append(dict(model=model,weather=d,cross_weather_mAP=value,clone_mAP=same,queries=len(ap),clone_queries=len(aps),all_positive_queries=trivial))
        output.append(dict(model=model,weather='mean',cross_weather_mAP=float(np.mean(values)),clone_mAP=same,queries=5895,clone_queries=len(aps),all_positive_queries=50))
    out=ROOT/'analysis/batch2/vkitti';csvout(out/'retrieval.csv',output)
    plot_tsne(models,output,selection,out)
    dump(out/'complete.json',dict(status='complete',same_protocol_as='ics_evidence part3',metadata_exact=True,
          test_scenes=selection['test_scenes'],cross_gallery='same scene, both cameras, clone abs frame difference<=10',
          clone_gallery='same scene, both cameras, abs frame difference4..30',positives='IoU-matched clone track, propagated only in clone',
          all_positive_galleries='included AP1',query_cap=None,training='none',single_training_seed=True))

def plot_tsne(models,metrics,selection,out):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    from sklearn.decomposition import PCA
    from sklearn.manifold import TSNE
    plt.rcParams.update({'font.size':10,'pdf.fonttype':42})
    for stage in ['stage4','stage1']:
        pair=[m for m in models if m.endswith(stage)];fig,axes=plt.subplots(1,2,figsize=(10,4.9));coords=[]
        if not pair:
            plt.close(fig);continue
        versions=['clone','fog','rain','sunset'];markers=['o','s','^','D']
        for ax,model in zip(axes,pair):
            ds=load(model);look={d:{int(a):i for i,a in enumerate(v['annotation_id'])} for d,v in ds.items()}
            bg=ds['clone']['Z'][[look['clone'][a] for a in selection['background_annotation_ids']]]
            points=np.array([ds[d]['Z'][look[d][r['annotations'][d]]] for r in selection['selected'] for d in versions])
            reduced=PCA(n_components=50,random_state=0).fit_transform(np.concatenate([bg,points]))
            estimator=TSNE(n_components=2,perplexity=30,init='pca',random_state=0,n_jobs=2);T=estimator.fit_transform(reduced);n=len(bg)
            ax.scatter(T[:n,0],T[:n,1],s=5,color='.75',alpha=.35,rasterized=True)
            for i,r in enumerate(selection['selected']):
                p=T[n+4*i:n+4*(i+1)];color=plt.get_cmap('tab10')(r['rank'])
                for j in range(1,4):ax.plot(p[[0,j],0],p[[0,j],1],color=color,lw=.5,alpha=.5)
                for j,d in enumerate(versions):
                    ax.scatter(p[j,0],p[j,1],s=30,marker=markers[j],color=color,edgecolors='white',linewidths=.25)
                    coords.append(dict(model=model,rank=r['rank'],scene=r['scene'],clone_instance_id=r['clone_instance_id'],weather=d,annotation_id=r['annotations'][d],x=float(p[j,0]),y=float(p[j,1])))
            m=next(r for r in metrics if r['model']==model and r['weather']=='mean')
            ax.set_title('%s\nFive-variant cross-weather mAP %.3f%%'%(model,m['cross_weather_mAP']));ax.set_xticks([]);ax.set_yticks([])
        fig.legend(handles=[Line2D([0],[0],marker=m,color='.4',ls='',label=v,ms=6) for m,v in zip(markers,versions)],loc='lower center',ncol=4,frameon=False)
        fig.suptitle('Fixed top10 paired VKITTI tracks; independent t-SNE per model; qualitative only')
        fig.tight_layout(rect=(0,.06,1,.94));f=ROOT/'figures/batch2';f.mkdir(parents=True,exist_ok=True)
        fig.savefig(str(f/('cross_weather_tsne_'+stage+'.pdf')),bbox_inches='tight');fig.savefig(str(f/('cross_weather_tsne_'+stage+'.png')),dpi=200,bbox_inches='tight');plt.close(fig)
        csvout(out/('tsne_'+stage+'_coordinates.csv'),coords)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--freeze-selection',action='store_true');a=p.parse_args()
    freeze_selection() if a.freeze_selection else main()
