"""Freeze test videos, GT boxes and t-SNE examples before any feature outcomes."""
import json,sys
from pathlib import Path
from collections import defaultdict,Counter
import numpy as np
from common import *

def main():
    protocols=[];night=None
    for domain in DOMAINS:
        source=RUN/'data/annotations'/(domain+'.json')
        data=json.loads(source.read_text());videos={v['id']:v for v in data['videos']}
        ordered=sorted(videos)
        fit=probe.split_mask(np.array([videos[v]['name'] for v in ordered]),0)
        test={v for v,is_fit in zip(ordered,fit) if not is_fit}
        assert test and len(test)<len(videos)
        by_video=defaultdict(list)
        for im in data['images']:by_video[im['video_id']].append(im)
        chosen=[];dose=[]
        for vid in sorted(test,key=lambda v:videos[v]['name']):
            selected=sorted(by_video[vid],key=lambda x:x['frame_id'])[::2][:30]
            chosen.extend(selected);dose.extend(i['id'] for i in selected[:5])
        ids={i['id'] for i in chosen};cat=next(c['id'] for c in data['categories'] if c['name'].lower()=='car')
        anns=[a for a in data['annotations'] if a['image_id'] in ids and a['category_id']==cat
              and not a.get('ignore',0) and not a.get('iscrowd',0) and a.get('instance_id',-1)>=0 and min(a['bbox'][2:])>0]
        subset=dict(data,images=chosen,annotations=anns,videos=[videos[v] for v in sorted(test)])
        dump(ROOT/'inputs'/('probe_'+domain+'.json'),subset)
        missing=[i['file_name'] for i in chosen if not (image_prefix(domain)/i['file_name']).is_file()]
        assert not missing,(domain,missing[:3])
        protocols.append(dict(domain=domain,annotation=str(source),annotation_sha256=sha(source),
             test_videos=[videos[v]['name'] for v in sorted(test)],frames=len(chosen),rois=len(anns),
             image_prefix=str(image_prefix(domain)),dose_image_ids=dose,original_frame_stride=2,frames_per_video=30,
             box_policy='all eligible GT car boxes, no size threshold or RoI cap'))
        if domain=='night':night=subset
    videos={v['id']:v['name'] for v in night['videos']};images={i['id']:i for i in night['images']}
    tracks=defaultdict(list)
    for a in night['annotations']:tracks[(videos[images[a['image_id']]['video_id']],a['instance_id'])].append(a)
    longest=sorted(tracks,key=lambda k:(-len(tracks[k]),k))[:10];selected=[]
    assert len(longest)==10 and all(len(tracks[k])>=4 for k in longest)
    for rank,key in enumerate(longest):
        aa=sorted(tracks[key],key=lambda a:(images[a['image_id']]['frame_id'],a['id']))
        for pos in np.linspace(0,len(aa)-1,4).round().astype(int):
            a=aa[pos];im=images[a['image_id']]
            selected.append(dict(rank=rank,video=key[0],instance_id=key[1],annotation_id=a['id'],image_id=im['id'],frame_id=im['frame_id'],bbox=a['bbox']))
    wanted={r['annotation_id'] for r in selected};imids={r['image_id'] for r in selected}
    extra=dict(night,images=[images[i] for i in sorted(imids)],annotations=[a for a in night['annotations'] if a['id'] in wanted])
    dump(ROOT/'inputs/tsne_night.json',extra)
    dump(ROOT/'manifests/tsne_selection.json',dict(rule='ten longest sampled Night test tracks, count descending then lexical identity tie-break; four uniform frames',
         selected=selected,views=['clean','dark_0.40','fog_0.45','fapk_0.50'],
         supplement='Dedicated extractions provide the three requested dose views at all four selected RoIs, including frames outside the first-five dose pool; never used in primary IR/retrieval/silhouette metrics'))
    dump(ROOT/'manifests/probe_samples.json',protocols)
    dump(ROOT/'manifests/analysis_protocol.json',dict(seed=0,bootstrap_repetitions=10000,bootstrap_unit='video, paired across compared models',
         tracking='Full-pool aggregate parity audit; paired inference uses split_mask seed0 test videos only; OVERALL primary, car supplementary',
         IR='sum paired normalized feature distances divided by sum nearest different-track clean distances, same eligible RoIs; omit single-identity frames only for IR',
         retrieval='P2 gap4 window30 forward (f+4,f+34], max4000 seed0 independently for each domain/pool and finite-view support; paired clean baseline for every dose support',
         silhouette='Exact cosine silhouette on clean/fap/photo plus all available first-five dose views; globally namespace tracks by video and instance; no feature subsampling',
         view_aggregation='equal mean over14 intervention views for summary bars; individual views all retained',
         C2_primary_controls=['C0','C4'],C1_C3='supplementary only',CI='pointwise percentile95%, 10000 replicates, not multiple-comparison adjusted',
         tracking_brightness='Original full-resolution grayscale mean of every annotated video frame; original full all-frame evaluation image pool',
         brightness_bins='Test Night within-domain thirds, stable sort brightness then video ID; batch2 also all four target domains pooled thirds',
         tsne='PCA50 then TSNE perplexity30 init=pca random_state0 per-model; qualitative display only',
         single_training_seed=1790355234,source_training_task='Read-only; second batch waits for training and frozen evaluations, or reports an externally stopped run'))
    print(json.dumps(protocols,ensure_ascii=False),flush=True)

if __name__=='__main__':main()
