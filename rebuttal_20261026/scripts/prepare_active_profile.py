#!/usr/bin/env python3
"""Find real-frame branch coverage without changing any confidence/sample gate."""
from profile_damot import *
install_fixed_fap_rng();torch.backends.cudnn.benchmark=False
frames=json.loads((OUT/'prototype_rebuild/overcast_frames.json').read_text())
allframes=json.loads((RUN/'manifests/overcast_frames.json').read_text());vids=defaultdict(list)
for r in allframes:vids[r['video_id']].append(r)
ref={}
for rs in vids.values():
    for i,r in enumerate(rs):ref[r['image_id']]=rs[min(i+1,len(rs)-1)]
def cache(rec):
    p=RUN/'cache/images/overcast'/rec['file_name'];p.parent.mkdir(parents=True,exist_ok=True)
    if not p.exists():shutil.copyfile(rec['source'],p)
    assert cv2.imread(str(p)) is not None
with concurrent.futures.ThreadPoolExecutor(8) as pool:list(pool.map(cache,[ref[r['image_id']] for r in frames]))
m,opt,ema,runner,stats=setup('FAP_ICS_APR');perframe=[];coverage=[]
orig=m.track_head._add_trimmed_pcl;current=[]
def capture(*a,**kw):
    current.extend(len(res.pos_bboxes) for res in a[3]);return orig(*a,**kw)
m.track_head._add_trimmed_pcl=capture
loader=loader_for(frames,'overcast',4,len(frames)//4,workers=4)
with torch.no_grad():
    for i,b in enumerate(loader):
        CTX['step']=i;fixed_seed(20261026+i);current.clear();loss=m.forward_train(**to_gpu(b))
        if len(current)!=4:
            coverage.append(dict(batch=i,positive_rois=list(current),sum=sum(current),note='native empty-pseudo drop; not used to rank individual frames'))
            print('COVERAGE',coverage[-1],flush=True);continue
        perframe+=list(zip(current,frames[4*i:4*i+4]));coverage.append(dict(batch=i,positive_rois=list(current),sum=sum(current)))
        print('COVERAGE',coverage[-1],flush=True)
ranked=sorted(perframe,key=lambda v:(-v[0],v[1]['image_id']))
# The first active-branch attempt used only four frames.  After the 20-step
# warm-up their positive-RoI count fell below the unchanged min_samples=96
# gate, so retain the failed trace and broaden the fixed real-frame coverage.
# The enlarged list is still selected only from native positive-RoI counts (no
# GT), and is shared by every history condition.
selected=[r for _,r in ranked[:32]]
plan=dict(domain='overcast',frames=selected,selection='32 distinct real frames selected by actual positive-RoI counts from the fixed 96-frame Overcast prototype collection list, no GT selection. Repeated fixed microprofile covers original ICS min_samples=96 after warm-up. This is not Night mean throughput.',coverage=coverage,top_positive_rois=[n for n,_ in ranked[:32]],confidence_gate=.7,min_samples=96,APR_score_gate=.75,previous_attempt='four-frame active profile failed after warm-up; retained profile_history0_failed trace and log')
dump(OUT/'active_profile_frame_manifest.json',plan)
del m,opt,ema,runner,loader;gc.collect();torch.cuda.empty_cache()
# Probe the highest-density four-frame batch repeatedly so the preflight tests
# the unchanged gate directly; the formal profiler itself uses all 32 selected
# real frames from the manifest.
probe=selected[:4]
m,opt,ema,runner,stats=setup('FAP_ICS_APR');loader=loader_for(probe,'overcast',4,3);it=iter(loader)
for i in range(3):train_step(m,opt,ema,runner,next(it),i)
torch.cuda.synchronize();stats['probe_frames']=[r['image_id'] for r in probe];dump(OUT/'active_profile_preflight.json',stats)
assert stats['successful_covariance_updates']>0,stats
assert stats['apr_loss_calls']>0,stats
print('ACTIVE PREFLIGHT PASS',stats,flush=True)
