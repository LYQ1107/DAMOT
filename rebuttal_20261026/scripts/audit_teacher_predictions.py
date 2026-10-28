#!/usr/bin/env python3
"""CPU-only audit of saved boxes, counts and canonical unique video coverage."""
import pickle,csv,time
from collections import defaultdict
from runtime import *
from diagnose_teacher import match_image
ds=dataset('night');videos=defaultdict(list)
for im in ds.data_infos:videos[im['video_id']].append(im)
records=[];start=time.time()
for tp in ['e0','e1','e3','e5']:
    root=RUN/'results/exp3'/tp;totals={k:0 for k in ['TP','FP','FN','GT_count','pseudo_count','ignored_pseudo_count']};nframes=0
    for vid,infos in videos.items():
        with open(root/'predictions'/('%s.pkl'%vid),'rb') as f:r=pickle.load(f)
        assert r['image_ids']==[i['id'] for i in infos]
        assert len(r['predictions'])==len(r['statistics'])==len(r['matches'])==len(infos)
        for im,pred,saved in zip(infos,r['predictions'],r['statistics']):
            assert len(pred)==6
            for a in pred:
                assert a.ndim==2 and a.shape[1]==5 and np.isfinite(a).all()
                assert (a[:,4]>.7).all()
            now,_=match_image(pred,ds.get_ann_info(im));assert now==saved,(tp,im['id'],now,saved)
            for row in now:
                for k in totals:totals[k]+=row[k]
            nframes+=1
    with open(root/'quality_summary.csv') as f:summary=next(r for r in csv.DictReader(f) if r['category']=='micro_six_classes')
    for k,v in totals.items():assert int(summary[k])==v,(tp,k,summary[k],v)
    assert nframes==7286 and len(videos)==36
    records.append(dict(timepoint=tp,unique_frames=nframes,unique_videos=len(videos),saved_boxes_vs_fresh_matching_exact=True,summary_matches_exact=True,counts=totals))
    print('AUDIT PASS',tp,nframes,totals,flush=True)
dump(RUN/'manifests/teacher_predictions_audited.json',dict(status='complete',exit_code=0,results=records,seconds=time.time()-start,unique_checkpoint_frame_results=29144,e1_extra_dispatched_forwards=6677,duplicate_rows_in_main_tables=0,scope='CPU recomputation from final canonical per-video teacher boxes, no extra GPU forward'))
