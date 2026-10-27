#!/usr/bin/env python3
"""Short, disposable training updates for resource measurement, never accuracy."""
import argparse,os,time,pickle,copy,types,logging,shutil,concurrent.futures,gc,subprocess,csv,io
from collections import defaultdict
from runtime import *
OUT=RUN/'results/exp4'
logging.getLogger('mmtrack').setLevel(logging.WARNING)
CTX={'step':0,'profile':False}

def exclusive_gpu_snapshot():
    device=os.environ.get('CUDA_VISIBLE_DEVICES','9')
    uuid=subprocess.check_output(['nvidia-smi','--id='+device,'--query-gpu=uuid','--format=csv,noheader'],text=True).strip()
    apps=list(csv.DictReader(io.StringIO(subprocess.check_output(['nvidia-smi','--query-compute-apps=gpu_uuid,pid,used_memory','--format=csv'],text=True))))
    peers=[dict((k.strip(),v.strip()) for k,v in r.items()) for r in apps if r['gpu_uuid'].strip()==uuid]
    assert not [r for r in peers if int(r['pid'])!=os.getpid()],('GPU acquired by another process; no process was terminated',peers)
    return dict(device=device,uuid=uuid,processes=peers)

def install_fixed_fap_rng():
    from darth.datasets.pipelines.transforms import SeqSPAFrequencyTransform
    orig=SeqSPAFrequencyTransform.__call__
    def call(self,results):
        state=np.random.get_state()
        np.random.seed(702100+CTX['step'])
        try:return orig(self,results)
        finally:np.random.set_state(state)
    SeqSPAFrequencyTransform.__call__=call

def local_selected(domain,n=96):
    frames=json.loads((RUN/'manifests'/(domain+'_frames.json')).read_text())
    chosen=[frames[int(i)] for i in np.linspace(0,len(frames)-1,n,dtype=int)]
    def one(rec):
        dest=RUN/'cache/images'/domain/rec['file_name'];dest.parent.mkdir(parents=True,exist_ok=True)
        if not dest.exists():shutil.copyfile(rec['source'],dest)
        im=cv2.imread(str(dest));assert im is not None and im.shape[:2]==(720,1280),dest
    with concurrent.futures.ThreadPoolExecutor(8) as ex:list(ex.map(one,chosen))
    dump(OUT/'prototype_rebuild'/(domain+'_frames.json'),chosen)
    return chosen

def frame_info(rec):
    return dict(id=rec['image_id'],video_id=rec['video_id'],frame_id=rec['frame_id'],file_name=rec['file_name'],height=720,width=1280)

class PairStream(torch.utils.data.Dataset):
    def __init__(self,records,domain,length):
        self.records=records;self.domain=domain;self.length=length
        allframes=json.loads((RUN/'manifests'/(domain+'_frames.json')).read_text())
        self.refs={}
        vids=defaultdict(list)
        for rec in allframes:vids[rec['video_id']].append(rec)
        for rs in vids.values():
            for i,r in enumerate(rs):self.refs[r['image_id']]=rs[min(i+1,len(rs)-1)]
    def __len__(self):return self.length
    def __getitem__(self,i):
        rec=self.records[i%len(self.records)];ref=self.refs[rec['image_id']]
        path=RUN/'cache/images'/self.domain/rec['file_name']
        rpath=RUN/'cache/images'/self.domain/ref['file_name']
        # References are real adjacent frames, although archived DARTH builds
        # its actual contrastive reference from key-frame appearance views.
        if not rpath.exists():rpath=Path(ref['source'])
        x,meta=preprocessed_frame(path,frame_info(rec),training_base=True)
        rx,rmeta=preprocessed_frame(rpath,frame_info(ref),training_base=True)
        return x[0],meta[0],rx[0],rmeta[0]

def collate_pair(items):
    return dict(img=torch.stack([i[0] for i in items]),img_metas=[i[1] for i in items],ref_img=torch.stack([i[2] for i in items]),ref_img_metas=[i[3] for i in items])

def loader_for(records,domain,batch,steps,workers=4):
    kwargs=dict(batch_size=batch,shuffle=False,num_workers=workers,collate_fn=collate_pair)
    if workers:
        kwargs.update(persistent_workers=True,prefetch_factor=2)
    return torch.utils.data.DataLoader(PairStream(records,domain,batch*steps),**kwargs)

def to_gpu(batch):
    batch['img']=batch['img'].cuda();batch['ref_img']=batch['ref_img'].cuda()
    n=len(batch['img_metas']);b=[torch.zeros((0,4),device='cuda') for _ in range(n)];l=[torch.zeros(0,dtype=torch.long,device='cuda') for _ in range(n)]
    batch.update(gt_bboxes=b,gt_labels=l,gt_match_indices=l,ref_gt_bboxes=b,ref_gt_labels=l)
    return batch

def tensor_bytes(obj):
    if isinstance(obj,torch.Tensor):return obj.numel()*obj.element_size()
    if isinstance(obj,dict):return sum(tensor_bytes(v) for v in obj.values())
    if isinstance(obj,(list,tuple)):return sum(tensor_bytes(v) for v in obj)
    return 0

def rebuild():
    histories={};storage=[];records=[]
    for i,domain in enumerate(['overcast','rainy','dawn_dusk','night'],1):
        dest=OUT/'prototype_rebuild'/(domain+'.pkl')
        if dest.exists():
            with open(dest,'rb') as f:rec=pickle.load(f)
            histories[domain]=rec['prototypes'];records.append(rec['record']);continue
        fixed_seed();frames=local_selected(domain)
        m,c=load_model('F_stage%d_e5'%i,deploy=False);m.train();m.teacher_detector.eval();m.teacher_track_head.eval();m.roi_proto_start_stage(domain)
        # Empty history during collection is deliberate: performance library
        # reconstruction is independent of any unrecorded original _history.
        mem=m.track_head.roi_proto_memory;used=[];pre=time.perf_counter()
        loader=loader_for(frames,domain,1,len(frames),workers=2)
        with torch.no_grad():
            for j,b in enumerate(loader):
                CTX['step']=j;fixed_seed(20261026+j)
                losses=m.forward_train(**to_gpu(b));used.append(frames[j])
                if sum(v.shape[0] for vs in mem._cand.values() for v in vs)>=1000:break
        torch.cuda.synchronize();collect_seconds=time.perf_counter()-pre
        candidate_bytes=tensor_bytes(mem._cand);candidate_score_bytes=tensor_bytes(mem._cand_scores)
        candidate_count=sum(v.shape[0] for vs in mem._cand.values() for v in vs)
        t=time.perf_counter();mem.finalize_stage(domain);finalize_ms=(time.perf_counter()-t)*1000
        assert domain in mem._history and len(mem._history[domain][1])>=20
        histories[domain]=mem._history[domain]
        rec=dict(domain=domain,checkpoint='F_stage%d_e5'%i,frames=len(used),image_ids=[r['image_id'] for r in used],candidate_count=candidate_count,CPU_candidate_features_bytes=candidate_bytes,CPU_candidate_scores_bytes=candidate_score_bytes,prototype_count=sum(len(v) for v in histories[domain].values()),CPU_history_bytes=tensor_bytes(histories[domain]),stage_finalize_ms=finalize_ms,collection_seconds=collect_seconds,score_thr=.75,class_label=1,topk=3,K_per_domain=100,min_candidates=20,feature_source='student sampled positive RoI embeddings via actual forward_train',exact_original_training_snapshot=False)
        save_pickle(dest,dict(prototypes=histories[domain],record=rec));records.append(rec)
        print('REBUILT',rec,flush=True)
        del m,mem,loader;gc.collect();torch.cuda.empty_cache()
    for n in range(5):
        chosen={d:histories[d] for d in ['overcast','rainy','dawn_dusk','night'][:n]}
        storage.append(dict(history_domains=n,prototype_count=sum(len(v) for dom in chosen.values() for v in dom.values()),CPU_history_bytes=tensor_bytes(chosen),CPU_history_MiB=tensor_bytes(chosen)/2**20,CPU_candidate_features_bytes=records[n-1]['CPU_candidate_features_bytes'] if n else 0,CPU_candidate_scores_bytes=records[n-1]['CPU_candidate_scores_bytes'] if n else 0,stage_finalize_ms=records[n-1]['stage_finalize_ms'] if n else 0,scope='per process; one process on V100; total equals per-process',source='measured reconstructed library, not original training snapshot'))
    save_pickle(OUT/'reconstructed_history.pkl',histories);write_csv(OUT/'prototype_storage_by_stage.csv',storage);dump(OUT/'prototype_rebuild/manifest.json',records)

def histories(n):
    with open(OUT/'reconstructed_history.pkl','rb') as f:h=pickle.load(f)
    return {d:h[d] for d in ['overcast','rainy','dawn_dusk'][:n]}

def instrument_activity(m):
    stats=dict(ics_calls=0,ics_effective_rois=[],centered_rois=[],covariance_calls=0,eigh_calls=0,successful_covariance_updates=0,apr_calls=0,apr_loss_calls=0,apr_selected_rois=0,view_shapes=[])
    h=m.track_head;bank=h.causal_bank;mem=h.roi_proto_memory
    def wrap(obj,name,fn):
        orig=getattr(obj,name)
        def call(*a,**kw):return fn(orig,a,kw)
        setattr(obj,name,call)
    def centered(orig,a,kw):
        stats['ics_calls']+=1;stats['ics_effective_rois'].append(int(a[0].shape[0]));out=orig(*a,**kw);stats['centered_rois'].append(0 if out[0] is None else int(out[0].shape[0]));return out
    wrap(h,'_build_proto_centered_feats',centered)
    def counter(key):
        def f(orig,a,kw):stats[key]+=1;return orig(*a,**kw)
        return f
    wrap(bank,'compute_covariance',counter('covariance_calls'));wrap(bank,'_eigh_top_pcs',counter('eigh_calls'))
    def covariance_update(orig,a,kw):
        result=orig(*a,**kw)
        if result:stats['successful_covariance_updates']+=1
        return result
    wrap(bank,'update_from_covariance',covariance_update)
    def apr(orig,a,kw):
        stats['apr_calls']+=1;r=orig(*a,**kw)
        if r is not None:stats['apr_loss_calls']+=1;stats['apr_selected_rois']+=int(mem.last_stats.get('selected_rois',0))
        return r
    wrap(mem,'compute_loss',apr)
    def views(orig,a,kw):
        r=orig(*a,**kw)
        stats['view_shapes'].append([list(v[0].shape) if v[0] is not None else None for v in r]);return r
    wrap(m,'_prepare_views',views)
    return stats

def setup(group,n_history=3):
    fixed_seed();m,c=load_model('F_stage3_e5',deploy=False);m.train();m.teacher_detector.eval();m.teacher_track_head.eval()
    m.log_view_stats=False;h=m.track_head
    if group=='baseline':
        m.student_pipeline.transforms=[t for t in m.student_pipeline.transforms if t.__class__.__name__!='SeqSPAFrequencyTransform']
    h.use_causal_direction_bank=group in ['FAP_ICS','FAP_ICS_APR']
    h.use_roi_proto_memory=group=='FAP_ICS_APR'
    c.model.track_head.use_causal_direction_bank=h.use_causal_direction_bank
    c.model.track_head.use_roi_proto_memory=h.use_roi_proto_memory
    if group=='baseline':c.model.student_pipeline=[v for v in c.model.student_pipeline if v['type']!='SeqSPAFrequencyTransform']
    c.dump(str(RUN/'configs'/('profiling_%s_history%d.py'%(group,n_history))))
    m.set_current_target_domain('night');m.roi_proto_start_stage('night')
    if h.use_roi_proto_memory:h.roi_proto_memory.load_state_dict_memory(histories(n_history))
    opt=torch.optim.SGD([p for p in m.parameters() if p.requires_grad],lr=c.optimizer.lr,momentum=c.optimizer.momentum,weight_decay=c.optimizer.weight_decay)
    from darth.core.hooks.ema import EMATrainHook
    ema=EMATrainHook(momentum=.002,interval=1,warm_up=100);runner=types.SimpleNamespace(model=m,iter=0);ema.before_run(runner)
    stats=instrument_activity(m)
    return m,opt,ema,runner,stats

def train_step(m,opt,ema,runner,b,step,profile=False):
    CTX['step']=step;fixed_seed(20261026+step);runner.iter=step
    rf=torch.profiler.record_function if profile else lambda name:__import__('contextlib').nullcontext()
    with rf('DAMOT::data_H2D'):b=to_gpu(b)
    opt.zero_grad(set_to_none=True)
    with rf('DAMOT::forward_and_loss'):losses=m.forward_train(**b);loss,log=m._parse_losses(losses)
    with rf('DAMOT::backward'):loss.backward()
    with rf('DAMOT::optimizer'):
        torch.nn.utils.clip_grad_norm_([p for p in m.parameters() if p.requires_grad],35.)
        opt.step()
    with rf('DAMOT::EMA'):ema.after_train_iter(runner)
    return {k:float(v) for k,v in log.items()}

def benchmark_records():
    frames=json.loads((RUN/'manifests/night_frames.json').read_text())
    chosen=[frames[int(i)] for i in np.linspace(0,len(frames)-1,512,dtype=int)]
    selection='512 uniform indices (initial environment/memory pilot)'
    # The archived min_samples=96 is large for batch4. Include a fixed density
    # stratum from the already-computed e0 predictions so covariance/eigh cost
    # is exercised by real RoIs, without GT-based selection or relaxed gates.
    if (RUN/'results/exp3/e0/complete.json').exists():
        counts={}
        for p in (RUN/'results/exp3/e0/predictions').glob('*.pkl'):
            with open(p,'rb') as f:pred=pickle.load(f)
            for im,boxes in zip(pred['image_ids'],pred['predictions']):counts[im]=sum(len(b) for b in boxes)
        assert len(counts)==7286
        dense=sorted(frames,key=lambda r:(-counts[r['image_id']],r['image_id']))[:64]
        uniform=[frames[int(i)] for i in np.linspace(0,len(frames)-1,448,dtype=int)]
        chosen=[]
        for k in range(16):chosen+=dense[4*k:4*k+4]+uniform[28*k:28*k+28]
        selection='64 densest frames by frozen e0 pseudo count >0.7, plus 448 uniform indices; blocks of 4 dense +28 uniform; no GT selection. Covers real ICS updates; this enriched workload is not a dataset-average throughput estimate.'
    # Ensure every input reference is on the same local cache; full Night was
    # decoded before any timing run and no NAS access occurs in the window.
    assert (RUN/'manifests/night_decode_validation.json').exists()
    dump(OUT/'adaptation_frame_manifest.json',dict(frames=chosen,selection=selection,input_pairs='key plus next frame from same video; actual DARTH contrastive reference is generated from key',base_resize=[1296,720],base_flip=False,seed_schedule='20261026 + step, reset each group/repeat; separate FAP RNG restored after transform',workers=4,cache='full Night local JPEG cache; OS page cache warmed by pilots',dtype='FP32'))
    return chosen

def choose_batch(records):
    path=OUT/'common_batch.json'
    import hashlib
    flow_hash=hashlib.sha256(json.dumps(records,sort_keys=True).encode()).hexdigest()
    if path.exists():
        old=json.loads(path.read_text())
        if old.get('frame_flow_sha256')==flow_hash:return old['batch_pairs']
        shutil.copyfile(path,OUT/('common_batch_before_flow_change_%d.json'%int(time.time())))
    failures=[]
    def pilot(bsize):
        m,opt,ema,runner,stats=setup('FAP_ICS_APR')
        loader=loader_for(records,'night',bsize,3);it=iter(loader)
        try:
            for i in range(3):train_step(m,opt,ema,runner,next(it),i)
            torch.cuda.synchronize()
            dump(OUT/('batch%d_pilot_activity_%s.json'%(bsize,flow_hash[:10])),dict(frame_flow_sha256=flow_hash,activity=stats,peak_allocated_MiB=torch.cuda.max_memory_allocated()/2**20,peak_reserved_MiB=torch.cuda.max_memory_reserved()/2**20))
        finally:
            del m,opt,ema,runner,loader,it;gc.collect();torch.cuda.empty_cache()
    for bsize in [4,2,1]:
        try:
            pilot(bsize);dump(path,dict(batch_pairs=bsize,frame_flow_sha256=flow_hash,pilots=failures,full_module_pilot='3 forward/backward/optimizer/EMA steps passed'))
            return bsize
        except RuntimeError as e:
            failures.append(dict(batch=bsize,error=repr(e)));dump(OUT/('batch%d_pilot_error.json'%bsize),failures[-1])
            if 'out of memory' not in str(e).lower():raise
            e.__traceback__=None
            gc.collect();torch.cuda.empty_cache()
    raise RuntimeError('No common batch fits: '+str(failures))

def pilot():
    records=benchmark_records();batch=choose_batch(records);print('COMMON BATCH',batch,flush=True)

def throughput():
    records=benchmark_records();batch=choose_batch(records);allrows=[]
    for group in ['baseline','FAP','FAP_ICS','FAP_ICS_APR']:
        for rep in range(3):
            dest=OUT/'raw_windows'/('%s_repeat%d.json'%(group,rep))
            if dest.exists():allrows.append(json.loads(dest.read_text())['summary']);continue
            m,opt,ema,runner,stats=setup(group);loader=loader_for(records,'night',batch,2000);it=iter(loader)
            for i in range(20):train_step(m,opt,ema,runner,next(it),i)
            torch.cuda.synchronize();torch.cuda.reset_peak_memory_stats();before={k:len(v) if isinstance(v,list) else v for k,v in stats.items()}
            hardware_before=os.popen('nvidia-smi --query-gpu=index,uuid,name,memory.used,utilization.gpu --format=csv').read()
            exclusive_before=exclusive_gpu_snapshot()
            load_before=os.getloadavg()
            start=time.perf_counter();losslogs=[];steps=100
            for i in range(20,120):losslogs.append(train_step(m,opt,ema,runner,next(it),i))
            torch.cuda.synchronize();elapsed=time.perf_counter()-start
            # A common 100-step window is normally >30s on V100. If it is not,
            # fail and uniformly extend every group rather than mix windows.
            if elapsed<30:raise RuntimeError('100-step window <30 seconds; extend all groups uniformly')
            allocated=torch.cuda.max_memory_allocated();reserved=torch.cuda.max_memory_reserved()
            exclusive_after=exclusive_gpu_snapshot()
            activity={k:(v[before[k]:] if isinstance(v,list) else v-before[k]) for k,v in stats.items()}
            if group=='FAP_ICS_APR':assert activity['apr_loss_calls']>0,activity
            row=dict(mode='adaptation',group=group,repeat=rep,batch_pairs=batch,history_domains=3 if group=='FAP_ICS_APR' else 0,warmup_steps=20,measured_steps=steps,window_seconds=elapsed,step_per_second=steps/elapsed,key_reference_pair_per_second=steps*batch/elapsed,key_frame_per_second=steps*batch/elapsed,model_association_FPS='',end_to_end_FPS='',peak_allocated_MiB=allocated/2**20,peak_reserved_MiB=reserved/2**20,FP32=True,gpu='Tesla V100-DGXS-32GB',conv_bn_fusion=False)
            finite=all(np.isfinite(v) for r in losslogs for v in r.values())
            dump(dest,dict(summary=row,activity=activity,ICS_coverage='covariance/eigh inactive in this Night batch4 window; not full active-estimator cost' if group in ['FAP_ICS','FAP_ICS_APR'] and not activity['covariance_calls'] else 'active or not applicable',losses=losslogs,finite_losses=finite,hardware_before=hardware_before,exclusive_GPU_before=exclusive_before,exclusive_GPU_after=exclusive_after,load_average_before=load_before,load_average_after=os.getloadavg(),timing_boundary='next(data) through H2D, real teacher/student/loss/backward/clip/SGD/EMA; synchronized only window boundaries; no profiler; excludes checkpoint IO and periodic saves'))
            assert finite,'non-finite profiling loss'
            allrows.append(row);write_csv(OUT/'adaptation_repeats.csv',allrows);print('THROUGHPUT',row,flush=True)
            del m,opt,ema,runner,loader,it;gc.collect();torch.cuda.empty_cache()
    dump(OUT/'adaptation_complete.json',dict(exit_code=0,windows=len(allrows),batch_pairs=batch))

def install_scopes(m):
    def wrap(obj,name,label):
        orig=getattr(obj,name)
        def call(*args,**kw):
            with torch.profiler.record_function('DAMOT::'+label):return orig(*args,**kw)
        setattr(obj,name,call)
    from darth.datasets.pipelines.transforms import SPAFrequencyTransform,SeqSPAFrequencyTransform
    import importlib
    pm=importlib.import_module('darth.models.utils.roi_proto_memory')
    wrap(pm,'roi_proto_loss_with_gate','APR.similarity_topk_loss')
    for obj,name,label in [(np.fft,'fft2','FAP.CPU_FFT'),(np.fft,'ifft2','FAP.CPU_iFFT'),(SPAFrequencyTransform,'_low_freq_mask','FAP.frequency_transform_inclusive'),(SeqSPAFrequencyTransform,'__call__','FAP.entire_augmentation_inclusive')]:wrap(obj,name,label)
    h=m.track_head;b=h.causal_bank;mem=h.roi_proto_memory
    for obj,name,label in [(h,'_add_trimmed_pcl','ICS.extra_association_and_loss_inclusive'),(h,'_build_proto_centered_feats','ICS.center_update'),(b,'compute_covariance','ICS.covariance'),(b,'update_from_covariance','ICS.covariance_EMA_and_eigh_inclusive'),(b,'_eigh_top_pcs','ICS.eigendecomposition'),(b,'trim','ICS.projection'),(mem,'collect','APR.collect_filter_CPU'),(mem,'compute_loss','APR.transfer_similarity_topk_loss_inclusive'),(mem,'get_history_prototypes','APR.history_CPU_concat')]:wrap(obj,name,label)
    for prefix,obj in [('teacher',m.teacher_detector),('student',m.detector)]:
        wrap(obj,'extract_feat','common_'+prefix+'.features')
        wrap(obj.rpn_head,'simple_test_rpn','common_'+prefix+'.RPN')
        wrap(obj.roi_head,'simple_test','common_'+prefix+'.RoI')
        wrap(obj.rpn_head,'forward','common_'+prefix+'.RPN_forward')
        wrap(obj.roi_head,'_bbox_forward','common_'+prefix+'.RoI_forward')
    wrap(m.track_head,'forward_train','common_student.association_inclusive')

def modules():
    plan=json.loads((OUT/'active_profile_frame_manifest.json').read_text());records=plan['frames'];domain=plan['domain'];batch=json.loads((OUT/'common_batch.json').read_text())['batch_pairs'];rows=[];storage=[]
    warmup=20;profile_steps=30
    dump(OUT/'module_window_selection.json',dict(warmup_steps=warmup,profile_steps=profile_steps,selection=plan['selection'],domain=domain,limitation='Separate active-branch microprofile on expanded fixed real Overcast frames; not the Night throughput workload. Original gates unchanged. The same 30-step profile length is used for every 0/1/2/3-history condition.'))
    # Add a transfer scope only in this separate profiler process. The original
    # expression and all math stay identical; throughput processes are untouched.
    import inspect,textwrap,importlib
    pm=importlib.import_module('darth.models.utils.roi_proto_memory')
    src=textwrap.dedent(inspect.getsource(pm.RoIPrototypeMemory.compute_loss))
    src=src.replace('        protos = protos.to(device, feats.dtype)',"        with torch.profiler.record_function('DAMOT::APR.history_transfer'):\n            protos = protos.to(device, feats.dtype)\n            self._profile_gpu_copy_bytes = protos.numel() * protos.element_size()")
    ns=pm.__dict__;exec(compile(src,'<profile-only APR transfer scope>','exec'),ns)
    pm.RoIPrototypeMemory.compute_loss=ns['compute_loss']
    for n in [ARGS.history]:
        m,opt,ema,runner,stats=setup('FAP_ICS_APR',n);loader=loader_for(records,domain,batch,warmup+profile_steps);it=iter(loader)
        for i in range(warmup):train_step(m,opt,ema,runner,next(it),i)
        install_scopes(m);torch.cuda.synchronize();torch.cuda.reset_peak_memory_stats();pre=copy.deepcopy(stats)
        with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU,torch.profiler.ProfilerActivity.CUDA],record_shapes=False,profile_memory=True,with_stack=False) as prof:
            for i in range(warmup,warmup+profile_steps):
                with torch.profiler.record_function('DAMOT::data_wait'):b=next(it)
                train_step(m,opt,ema,runner,b,i,True);prof.step()
            torch.cuda.synchronize()
        prof.export_chrome_trace(str(OUT/('profile_history%d.json'%n)))
        for e in prof.key_averages():
            if e.key.startswith('DAMOT::'):
                rows.append(dict(history_domains=n,profile_domain=domain,workload='active_branch_microbenchmark_repeated_four_real_frames',scope=e.key,calls=e.count,profile_steps=profile_steps,CPU_total_ms_per_step=e.cpu_time_total/1000/profile_steps,CPU_self_ms_per_step=e.self_cpu_time_total/1000/profile_steps,GPU_total_ms_per_step=e.cuda_time_total/1000/profile_steps,GPU_self_ms_per_step=e.self_cuda_time_total/1000/profile_steps,nesting='inclusive totals overlap; do not add nested totals; CUDA kernels async with CPU',worker_execution='FAP executes in main-process _prepare_views, not loader workers'))
        mem=m.track_head.roi_proto_memory
        assert stats['covariance_calls']>pre['covariance_calls'] and stats['eigh_calls']>pre['eigh_calls'],'Extend the same fixed profiler window: ICS did not update'
        assert stats['successful_covariance_updates']>pre['successful_covariance_updates'],'Covariance/eigh must successfully update PCs'
        if n:assert stats['apr_loss_calls']>pre['apr_loss_calls']
        storage.append(dict(history_domains=n,profile_domain=domain,workload='active_branch_microbenchmark_repeated_four_real_frames',CPU_history_bytes=tensor_bytes(mem._history),CPU_candidate_bytes=tensor_bytes(mem._cand),CPU_candidate_score_bytes=tensor_bytes(mem._cand_scores),GPU_prototype_copy_bytes=getattr(mem,'_profile_gpu_copy_bytes',0),GPU_peak_allocated_MiB=torch.cuda.max_memory_allocated()/2**20,GPU_peak_reserved_MiB=torch.cuda.max_memory_reserved()/2**20,APR_loss_active_calls=stats['apr_loss_calls']-pre['apr_loss_calls'],ICS_covariance_calls=stats['covariance_calls']-pre['covariance_calls'],ICS_eigh_calls=stats['eigh_calls']-pre['eigh_calls'],ICS_successful_updates=stats['successful_covariance_updates']-pre['successful_covariance_updates']))
        dump(OUT/('module_activity_history%d.json'%n),stats)
        write_csv(OUT/('module_time_history%d.csv'%n),rows);write_csv(OUT/('prototype_runtime_history%d.csv'%n),storage)
        print('MODULES',n,storage[-1],flush=True)
        # Class-level scope wrappers must not nest across n conditions.
        # Restored by a fresh process for each n in the CLI below.
        del m,opt,ema,runner,loader,it,prof;gc.collect();torch.cuda.empty_cache()
        break

def inference():
    allframes=json.loads((RUN/'manifests/night_frames.json').read_text());vids=defaultdict(list)
    for r in allframes:vids[r['video_id']].append(r)
    # Preselected first eight complete videos by numeric id, independent of GT.
    records=sum([vids[k] for k in sorted(vids)[:8]],[]);assert len(records)>=1500
    dump(OUT/'inference_frame_manifest.json',dict(frames=records,selection='first eight complete Night videos by id; fixed before timing',batch=1,FP32=True,fusion=True))
    m,c=load_model('F_stage4_e5',deploy=True,fusion=True);rows=[]
    with torch.no_grad():
        for r in records[:20]:
            x,meta=preprocessed_frame(RUN/'cache/images/night'/r['file_name'],frame_info(r));m.simple_test(x.cuda(),meta,rescale=True)
        for rep in range(3):
            # Model+association uses all preprocessed tensors on CPU page cache;
            # CUDA boundaries isolate each complete-video inference window.
            prepared=[preprocessed_frame(RUN/'cache/images/night'/r['file_name'],frame_info(r)) for r in records]
            for mode in ['model_association','end_to_end']:
                m.tracker.reset();torch.cuda.synchronize();torch.cuda.reset_peak_memory_stats();start=time.perf_counter();times=0.
                for j,r in enumerate(records):
                    x,meta=prepared[j] if mode=='model_association' else preprocessed_frame(RUN/'cache/images/night'/r['file_name'],frame_info(r))
                    x=x.cuda()
                    if mode=='model_association':torch.cuda.synchronize();t=time.perf_counter()
                    m.simple_test(x,meta,rescale=True)
                    if mode=='model_association':torch.cuda.synchronize();times+=time.perf_counter()-t
                torch.cuda.synchronize();elapsed=time.perf_counter()-start
                measured=times if mode=='model_association' else elapsed
                row=dict(mode='inference_'+mode,group='deployed_student',repeat=rep,batch_pairs=1,history_domains=0,warmup_steps=20,measured_steps=len(records),window_seconds=measured,step_per_second='',key_reference_pair_per_second='',key_frame_per_second='',model_association_FPS=len(records)/measured if mode=='model_association' else '',end_to_end_FPS=len(records)/measured if mode=='end_to_end' else '',peak_allocated_MiB=torch.cuda.max_memory_allocated()/2**20,peak_reserved_MiB=torch.cuda.max_memory_reserved()/2**20,FP32=True,gpu='Tesla V100-DGXS-32GB',conv_bn_fusion=True)
                rows.append(row);write_csv(OUT/'inference_repeats.csv',rows);print('INFERENCE',row,flush=True)
            del prepared;gc.collect()
    dump(OUT/'inference_complete.json',dict(exit_code=0,frames_per_window=len(records),complete_videos=8,repeats=3,FAP='not in deployment forward',ICS='not in deployment forward',APR='not in deployment forward'))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('mode',choices=['rebuild','pilot','throughput','modules','inference']);p.add_argument('--history',type=int,default=3,choices=range(4));args=p.parse_args();ARGS=args
    install_fixed_fap_rng();torch.backends.cudnn.benchmark=False
    dump(OUT/('job_'+args.mode+'.json'),dict(pid=os.getpid(),gpu=os.environ.get('CUDA_VISIBLE_DEVICES'),argv=sys.argv,start=time.time()))
    globals()[args.mode]()
