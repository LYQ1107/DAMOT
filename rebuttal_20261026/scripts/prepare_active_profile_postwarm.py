#!/usr/bin/env python3
"""Select fixed active-branch frames after the real profiling warm-up.

Performance-only coverage probe: no GT and no change to confidence or
min_samples gates. The model is warmed through the same forward/backward/
SGD/EMA path as profiling, then held fixed while real Overcast frames are
ranked by native positive RoI counts.
"""
from profile_damot import *

install_fixed_fap_rng()
torch.backends.cudnn.benchmark = False

def cache_records(records):
    def one(rec):
        dest = RUN/'cache/images/overcast'/rec['file_name']
        dest.parent.mkdir(parents=True, exist_ok=True)
        if not dest.exists():
            shutil.copyfile(rec['source'], dest)
        im = cv2.imread(str(dest))
        assert im is not None and im.shape[:2] == (720, 1280), dest
    with concurrent.futures.ThreadPoolExecutor(8) as ex:
        list(ex.map(one, records))

def scan(model, records, tag):
    captured=[]
    orig=model.track_head._add_trimmed_pcl
    def capture(*args, **kwargs):
        captured.extend(len(res.pos_bboxes) for res in args[3])
        return orig(*args, **kwargs)
    model.track_head._add_trimmed_pcl=capture
    vals=[]
    loader=loader_for(records, 'overcast', 4, max(1, len(records)//4), workers=0)
    with torch.no_grad():
        for i,b in enumerate(loader):
            CTX['step']=i; fixed_seed(20261026+i); captured.clear()
            model.forward_train(**to_gpu(b))
            if len(captured)==4:
                vals.extend(zip(captured, records[4*i:4*i+4]))
    model.track_head._add_trimmed_pcl=orig
    dump(OUT/('postwarm_coverage_'+tag+'.json'), dict(
        tag=tag, frames=len(records), usable=len(vals),
        max_batch_sum=max((sum(v for v,_ in vals[i:i+4])
                           for i in range(0,len(vals)-3,4)), default=0),
        values=[dict(positive_rois=int(v),image_id=r['image_id'])
                for v,r in vals]))
    return vals

allframes=json.loads((RUN/'manifests/overcast_frames.json').read_text())
base=json.loads((OUT/'active_profile_frame_manifest.json').read_text())['frames']
candidate=base
cache_records(candidate)
m,opt,ema,runner,stats=setup('FAP_ICS_APR')
loader=loader_for(candidate,'overcast',4,20,workers=4);it=iter(loader)
for i in range(20):
    train_step(m,opt,ema,runner,next(it),i)
m.eval()
vals=scan(m,candidate,'32')
def max_sum(vs):
    return max((sum(v for v,_ in vs[i:i+4])
                for i in range(0,len(vs)-3,4)), default=0)
if max_sum(vals)<96:
    candidate=[allframes[int(i)] for i in np.linspace(0,len(allframes)-1,512,dtype=int)]
    cache_records(candidate)
    vals=scan(m,candidate,'512')
assert max_sum(vals)>=96, {'max_batch_sum':max_sum(vals),
                           'candidate_frames':len(candidate)}
ranked=sorted(vals,key=lambda z:(-int(z[0]),int(z[1]['image_id'])))
selected=[r for _,r in ranked[:32]]
plan=dict(
    domain='overcast', frames=selected,
    selection='32 distinct real Overcast frames ranked after the same 20-step forward/backward/SGD/EMA warm-up used by module profiling; post-warm-up model then held fixed for native positive-RoI scan; no GT selection.',
    top_positive_rois=[int(v) for v,_ in ranked[:32]],
    confidence_gate=.7, min_samples=96, APR_score_gate=.75,
    coverage_probe='postwarm_coverage_32.json' if len(candidate)==32 else 'postwarm_coverage_512.json',
    candidate_pool_frames=len(candidate),
    previous_failures=['four-frame prewarm scan: no update after warm-up',
                       '32-frame prewarm scan: no update after warm-up'])
dump(OUT/'active_profile_frame_manifest.json',plan)
probe=selected[:4]
del m,opt,ema,runner,loader
gc.collect(); torch.cuda.empty_cache()
m,opt,ema,runner,stats=setup('FAP_ICS_APR')
loader=loader_for(probe,'overcast',4,3,workers=0);it=iter(loader)
for i in range(3):
    train_step(m,opt,ema,runner,next(it),i)
torch.cuda.synchronize()
stats['probe_frames']=[r['image_id'] for r in probe]
dump(OUT/'active_profile_preflight.json',stats)
assert stats['successful_covariance_updates']>0,stats
assert stats['apr_loss_calls']>0,stats
print('POSTWARM ACTIVE PREFLIGHT PASS',dict(
    candidate_pool_frames=len(candidate),
    top_positive_rois=plan['top_positive_rois'][:8], stats=stats), flush=True)
