"""Exact original test split and equal-video paired bootstrap, 10000 seed0."""
from common import *
from statistics_utils import paired_summary
DOMAINS=['overcast','rainy','dawn_dusk','night']
details=[];effects=[]
for domain in DOMAINS:
    new=readcsv(ROOT/'tracking/NoICS'/domain/'per_video.csv')
    old=readcsv(ROOT/'tracking/HF'/domain/'per_video.csv')
    aa={(r['video'],r['category']):r for r in new}
    bb={(r['video'],r['category']):r for r in old};assert aa.keys()==bb.keys()
    videos=sorted({r['video'] for r in new})
    fit=dict(zip(videos,probe.split_mask(np.array(videos),0)))
    for key in sorted(aa):
        a,b=aa[key],bb[key]
        assert a['test']==b['test']==str(not bool(fit[a['video']]))
        assert a['frames']==b['frames']
        if a['test']!='True':continue
        r=dict(model='NoICS',reference='HF',domain=domain,video=a['video'],video_id=a['video_id'],category=a['category'],test=True,frames=int(a['frames']))
        for metric in ['HOTA','AssA','IDF1']:
            r['NoICS_'+metric]=float(a[metric]);r['HF_'+metric]=float(b[metric]);r[metric+'_delta']=float(a[metric])-float(b[metric])
        details.append(r)
for domain in DOMAINS+['all_four_domains_pooled']:
    for category in ['OVERALL','car']:
        rr=[r for r in details if r['category']==category and (r['domain']==domain or domain=='all_four_domains_pooled')]
        for metric in ['HOTA','AssA','IDF1']:
            effects.append(dict(model='NoICS',reference='HF',domain=domain,category=category,metric=metric,**paired_summary([r[metric+'_delta'] for r in rr])))
out=ROOT/'results/video_pairs'
csvout(out/'video_differences.csv',details);csvout(out/'paired_summary.csv',effects)
dump(out/'complete.json',dict(status='complete',single_training_seed=True,training_seed=1790355234,
    test_only=True,split='exact ics_advantage video-name MD5 split, seed0',bootstrap_repetitions=10000,bootstrap_seed=0,
    estimand='equal-video paired mean difference; median is a point estimate',CI='pointwise percentile95% interval of mean differences; no multiplicity correction',
    pooled_statistic='equal-video pool; distinct from the main four-domain macro average and from nonlinear official aggregates',
    test_videos_by_domain={d:len({r['video'] for r in details if r['domain']==d}) for d in DOMAINS}))
print('VIDEO_PAIR_BOOTSTRAP_COMPLETE',len(details),len(effects),flush=True)
