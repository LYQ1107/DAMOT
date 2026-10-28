#!/usr/bin/env python3
import json,csv,sys,math,statistics,hashlib
from pathlib import Path
RUN=Path(__file__).resolve().parents[1]
def read(p):
    with open(p) as f:return list(csv.DictReader(f))
def write(p,rows):
    with open(p,'w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
def exp2():
    root=RUN/'results/exp2';rows=[]
    for stage,m in enumerate(['S','F_stage1_e5','F_stage2_e5','F_stage3_e5','F_stage4_e5']):
        c=json.loads((root/m/'complete.json').read_text());assert c['exit_code']==0 and c['frames']==9103 and c['videos']==45
        summ={r['']:r for r in read(root/m/'summary_metrics.csv')};track={r['']:r for r in read(root/m/'track_metrics.csv')}
        for cls in ['OVERALL','car']:
            r=dict(model=m,stage=stage,epoch=20 if m=='S' else 5,dataset='clear_holdout_45v',category=cls,frames=9103,videos=45)
            r.update({k+'_percent':float(summ[cls][k]) for k in ['HOTA','IDF1','AssA','DetA']});r['IDSw_count']=int(track[cls]['IDSw'])
            source=next((a for a in rows if a['model']=='S' and a['category']==cls),r)
            r.update({'delta_'+k+'_pp':r[k+'_percent']-source[k+'_percent'] for k in ['HOTA','IDF1','AssA','DetA']})
            r.update(checkpoint=str(RUN/'cache/weights'/(m+'.pth')),detector_score_thr=.05,tracker_init_score_thr=.30,tracker_obj_score_thr=.20,student_native_QDTrack=True,conv_bn_fusion=True,provenance='new_frozen_student_inference_and_official_evaluation')
            rows.append(r)
    write(root/'clear_retention.csv',rows)
    p=root/'clear_holdout_manifest.json';a=json.loads(p.read_text());a.update(frame_manifest=str(RUN/'manifests/clear_holdout_45v_frames.json'),image_sha256_manifest=str(RUN/'manifests/clear_holdout_45v_image_hashes.json'),all_frames_decoded=True,video_overlap_with_source_or_target=[],protocol='frozen epoch5 only; no adaptation, tuning or checkpoint selection on holdout GT');p.write_text(json.dumps(a,indent=2,ensure_ascii=False))
    return rows
def exp3():
    root=RUN/'results/exp3';outs={name:[] for name in ['summary','per_class','per_video']}
    gt=None
    for tp in ['e0','e1','e3','e5']:
        c=json.loads((root/tp/'complete.json').read_text());assert c['exit_code']==0 and c['frames']==7286 and c['videos']==36
        for name in outs:
            rows=read(root/tp/('quality_'+name+'.csv'))
            for r in rows:
                assert int(r['TP'])+int(r['FN'])==int(r['GT_count'])
                assert int(r['TP'])+int(r['FP'])+int(r['ignored_pseudo_count'])==int(r['pseudo_count'])
            outs[name].extend(rows)
        cur=int(outs['summary'][-2]['GT_count'])
        assert gt is None or gt==cur;gt=cur
    write(root/'teacher_pseudo_quality.csv',outs['summary']);write(root/'teacher_pseudo_quality_per_class.csv',outs['per_class']);write(root/'teacher_pseudo_quality_per_video.csv',outs['per_video'])
    (root/'filtering_and_coordinates.txt').write_text('Frozen teacher_detector branch, strict state load. e0 is F stage03 epoch5 teacher (not Clear source).\nTeacher extract_feat -> test RPN -> RoI simple_test with original teacher test_cfg: score_thr=.05, NMS IoU=.5, max_per_img=100; original filter uses strict score > .7. This happens before teacher_pipeline geometry.\nAll 7286 Night frames at all four checkpoints, same ordered manifest. Deterministic base resize 1296x720 keep ratio, no flip; SeqNormalize and SeqPad conventions. Predicted xyxy divided by actual scale_factor to original GT coordinates. Separate parity preflight agrees with real _prepare_views within 0.001px and 1e-6 confidence, with identical filtered class/count.\nSix classes via actual dataset.cat2label. GT from dataset.get_ann_info; invalid/ignore excluded, crowd in bboxes_ignore. Same-class greedy confidence-ordered one-to-one IoU>=.5; repeated predictions are FP; unmatched predictions inside any crowd region at IoF>.5 are ignored.\nPrecision TP/(TP+FP), recall TP/(TP+FN), zero denominator N/A. pseudo_count includes ignored pseudo detections; ignored_pseudo_count is explicit.\nGT used only for post-hoc matching; no GT in teacher inference, filtering or sample selection; no backward or EMA.\nStable/improved trajectory cannot rule out every error-propagation mechanism.\n')
    return outs['summary']
def exp4():
    root=RUN/'results/exp4';a=json.loads((root/'adaptation_complete.json').read_text());b=json.loads((root/'inference_complete.json').read_text());assert a['windows']==12 and b['repeats']==3
    raw=read(root/'adaptation_repeats.csv')+read(root/'inference_repeats.csv');summary=[]
    keys=['window_seconds','step_per_second','key_reference_pair_per_second','key_frame_per_second','model_association_FPS','end_to_end_FPS']
    groups=[]
    for r in raw:
        key=(r['mode'],r['group'])
        if key not in groups:groups.append(key)
    for mode,group in groups:
        rs=[r for r in raw if (r['mode'],r['group'])==(mode,group)];assert len(rs)==3
        r={k:rs[0][k] for k in ['mode','group','batch_pairs','history_domains','warmup_steps','measured_steps','FP32','gpu','conv_bn_fusion']};r['repetitions']=3
        for k in keys:
            vals=[float(t[k]) for t in rs if t[k]!=''];r[k+'_mean']=statistics.mean(vals) if vals else '';r[k+'_sample_std']=statistics.stdev(vals) if len(vals)>1 else ''
        r['peak_allocated_MiB_max']=max(float(t['peak_allocated_MiB']) for t in rs);r['peak_reserved_MiB_max']=max(float(t['peak_reserved_MiB']) for t in rs)
        if mode=='adaptation':
            activities=[json.loads((root/'raw_windows'/('%s_repeat%d.json'%(group,int(t['repeat'])))).read_text())['activity'] for t in rs]
            r['ICS_covariance_calls_total']=sum(t['covariance_calls'] for t in activities);r['ICS_eigh_calls_total']=sum(t['eigh_calls'] for t in activities)
            r['ICS_coverage_note']='inactive covariance/eigh in Night batch4; not full active-estimator cost' if group in ['FAP_ICS','FAP_ICS_APR'] and not r['ICS_covariance_calls_total'] else 'not applicable or active'
        else:r.update(ICS_covariance_calls_total='',ICS_eigh_calls_total='',ICS_coverage_note='not in deployed forward')
        summary.append(r)
    write(root/'efficiency_summary.csv',summary)
    modules=[];storage=[]
    for n in range(4):modules+=read(root/('module_time_history%d.csv'%n));storage+=read(root/('prototype_runtime_history%d.csv'%n))
    write(root/'module_time_breakdown.csv',modules);write(root/'prototype_runtime_storage.csv',storage)
    for n,r in enumerate(storage):
        assert int(r['history_domains'])==n
        if n:assert int(r['APR_loss_active_calls'])>0
        else:assert int(r['APR_loss_active_calls'])==0
    (root/'timing_boundaries.txt').write_text('Hardware: one Tesla V100-DGXS-32GB, CUDA_VISIBLE_DEVICES=9, FP32, cuDNN autotune disabled consistently (the initial autotune-enabled baseline measurements were superseded and retained). No other task GPU job ran on device9 during formal timing; snapshots retained. Background machine load and CPU details are in manifests. This is not original four-A100 speed.\nFour controlled computation groups share F stage03 epoch5 weights and teacher/ICS buffers; reset before every group/repeat with fresh identical SGD state and identical real input/augmentation random schedule. Static disabled-module buffers remain allocated in all switch groups. These short updates are disposable and are never accuracy checkpoints.\nAdaptation: 20 warm-up, 100 timed steps, 3 repetitions; each window >=30s. Main throughput has no torch profiler. Includes DataLoader wait, local JPEG decode/base resize, reference read, H2D, CPU NumPy FAP in main _prepare_views (not loader workers), both teacher passes, student views, losses, backward, gradient clipping, SGD, EMA. Periodic checkpoint saving is excluded. Explicit CUDA synchronization only at window boundaries; original model loss/stat .item() calls remain.\nInputs: raw key and adjacent reference 1280x720 (1296x720 keep-ratio base), batch pairs from common_batch.json. Archived DARTH constructs its actual contrastive view from key and does not consume real ref_img after input; key/reference input IO/H2D is still included. Teacher/student/contrastive augmented tensor sizes are logged per step in raw_windows activity.view_shapes. Worker count=4, full locally cached Night, same 512-key frame flow for all groups (448 uniform +64 high-density frames selected solely by frozen e0 pseudo counts, interleaved 28+4; no GT selection; density stratum retained, but actual Night batch4 never reached min_samples=96 in the first complete window; not a full active-estimator cost or dataset-average throughput estimate).\nThroughput units: step/s; key-reference input pairs/s = batch*step/s; key frames/s same number. Multiple augmented views are not called ordinary FPS. Peak allocated/reserved resets after warm-up, includes live teacher+student, gradients, momentum and temporary tensors.\nModule profiling is separate: 20 warm-up +10 profiled steps for each 0/1/2/3 history condition on a separate fixed real Overcast dense-frame microbenchmark, because Night batch4 did not reach min_samples96 (module_window_selection.json). CPU and GPU total/self milliseconds per step are provided. Inclusive nested scopes overlap; do not sum parent and child times. FAP.frequency_transform CPU self excludes nested FFT/iFFT and represents amplitude/phase/mask/other frequency operations; entire_augmentation includes casting/clipping. ICS extra association is trimmed match/loss on existing features, not an extra backbone pass. Common teacher cost is never attributed wholly to ICS.\nAPR transfer scope wraps the exact original tensor .to expression only in separate profiling processes. Similarity/top-k/loss and CPU history concat/collection scopes are separate. APR reconstruction uses exact collect/finalize filters and real positive RoI features. Libraries are measured reconstructions, not unpersisted original snapshots. Per-domain prototype and candidate tensor bytes are measured numel*element_size; scores separate; Python container overhead excluded. GPU copy bytes are payload per transient concatenated history matrix, full GPU peak also recorded. Single process, so process total equals run total.\nDeployment: native student QDTrack with Conv-BN fusion, batch1, first eight complete fixed Night videos, 20 warm-up frames, each repeat starts/reset tracker and frame0 also resets. Model+association FPS times only simple_test after H2D synchronization; excludes read/preprocess/H2D. End-to-end includes JPEG read, resize/normalize/pad, H2D and tracking. All selected videos evaluated in chronological order three times. FAP, ICS, APR do not execute in this deployed forward; no per-module inference FPS fabricated.\n')
    return summary
def summary_all():
    a=read(RUN/'results/exp1/apr_forgetting_metrics.csv');b=exp2();c=exp3();d=exp4();text=[]
    text.append('DAMOT rebuttal 四项补充实验真实执行结果\nRUN_ROOT: '+str(RUN)+'\n')
    text.append('实验1：N_old、F_old、F各16个阶段-域单元共48个。完整精度CLEAR IDF1/IDSw直接复用；HOTA/DetA/AssA使用已有合格预测统一重跑评估器，修正无GT图像遗漏索引的问题；旧口径CSV单独保留。重新计算遗忘及GT归一化IDSw，未提取身份特征、未新增旧域模型前向。FinalOld均来自Night epoch5。目标域沿用适应池，不是独立目标泛化集。')
    for r in a:
        if r['category']=='OVERALL' and r['metric'] in ['IDF1_percent','AssA_percent']:text.append('%s %s: FinalOld=%s%%，AvgF=%s个百分点。'%(r['model'],r['metric'],r['FinalOld_percent'],r['AvgF_percentage_points']))
    final=read(RUN/'results/exp1/apr_old_domain_metrics.csv')
    for m in ['N_old','F_old','F']:
        rs=[r for r in final if r['model']==m and r['category']=='OVERALL'];n=sum(int(r['GT_count']) for r in rs);ids=sum(int(r['IDSw_count']) for r in rs);text.append('%s三旧域IDSw合计=%d，GT=%d，合并计数IDSw/1000GT=%.12g。'%(m,ids,n,1000*ids/n))
    text.append('旧版比较同为global_pc_ema，但每GPU batch16/14、归档gpu_ids对应4/6卡（名义总batch64/84）等不同，不是严格只差APR；F更换ICS估计器，仅单列参考。指标改善方向不一致，不能全部归因APR。种子只记录，不分析、不重训。')
    text.append('\n实验2：五模型全部新做冻结学生前向及官方评估。clear_holdout_45v为45视频9103帧，指定SHA256核验、与源/目标视频无交集。只评估固定源epoch20/适应epoch5，不使用holdout选参。')
    for r in b:
        if r['category']=='OVERALL':text.append('%s: HOTA=%.12g%%，IDF1=%.12g%%，AssA=%.12g%%；相对S差值 %.12g / %.12g / %.12g 个百分点。'%(r['model'],r['HOTA_percent'],r['IDF1_percent'],r['AssA_percent'],r['delta_HOTA_pp'],r['delta_IDF1_pp'],r['delta_AssA_pp']))
    text.append('\n实验3：e0/e1/e3/e5教师各7286帧36视频，共29144个唯一时点-图像结果；e1调度失误额外重复6677帧前向（错误/进程日志保留），主表无重复，最终逐帧伪框与匹配经独立CPU重算核验。e0是完成Dawn-Dusk后的教师。真实teacher_detector流程、score>0.7，GT仅事后匹配。')
    micro=[r for r in c if r['category']=='micro_six_classes']
    for r in micro:text.append('%s: TP=%s，FP=%s，FN=%s，GT=%s，pseudo=%s；precision=%s，recall=%s，TP/frame=%s，FP/frame=%s。'%(r['timepoint'],r['TP'],r['FP'],r['FN'],r['GT_count'],r['pseudo_count'],r['precision_ratio'],r['recall_ratio'],r['TP_per_frame'],r['FP_per_frame']))
    pr=[float(r['precision_ratio']) for r in micro];rc=[float(r['recall_ratio']) for r in micro];fp=[int(r['FP']) for r in micro]
    text.append('precision是否每个相邻时点都下降：%s；recall是否每个相邻时点都下降：%s；FP是否每个相邻时点都上升：%s。该轨迹不能证明所有错误传播机制均被排除。'%(all(x>y for x,y in zip(pr,pr[1:])),all(x>y for x,y in zip(rc,rc[1:])),all(x<y for x,y in zip(fp,fp[1:]))))
    text.append('\n实验4：V100单卡FP32控制计算开关成本，不是精度消融。实际batch见表，各组相同；同起点重置，真实forward/backward/SGD/EMA，20步预热、100步计时、3次重复；周期存盘不计。')
    for r in d:
        if r['mode']=='adaptation':text.append('%s: %.12g step/s，%.12g pair/s，峰值allocated %.12g MiB，reserved %.12g MiB。'%(r['group'],r['step_per_second_mean'],r['key_reference_pair_per_second_mean'],r['peak_allocated_MiB_max'],r['peak_reserved_MiB_max']))
        else:text.append('%s: model+association=%s FPS，end-to-end=%s FPS，峰值allocated=%s MiB。'%(r['mode'],r['model_association_FPS_mean'],r['end_to_end_FPS_mean'],r['peak_allocated_MiB_max']))
    text.append('性能帧流含448个均匀索引帧和64个按冻结e0伪框数选择的密集帧，原意覆盖高密度场景，但实际Night batch4有效RoI仍低于ICS min_samples=96，协方差/特征分解未触发。主表是这种门控条件下的实际成本，不能当作原训练完整ICS成本；另以四个真实Overcast密集帧复放测量可触发时的完整模块成本，单独标注数据和调用次数，不合并为Night吞吐。不使用GT选帧，不降低置信度或样本阈值。各组共用此固定流，因此这是给定工作负载的成本比较，不是整域平均适应吞吐。cuDNN autotune统一关闭；启用autotune的预备记录已另存，不参与正式均值。')
    proto=read(RUN/'results/exp4/prototype_storage_by_stage.csv')
    for r in proto:
        text.append('历史%d域：原型%s个，CPU历史张量%s bytes (%s MiB)；对应阶段finalize=%s ms。候选张量快照是在该stage finalize之前、历史库快照在之后，不能直接相加为同一时刻峰值。'%(int(r['history_domains']),r['prototype_count'],r['CPU_history_bytes'],r['CPU_history_MiB'],r['stage_finalize_ms']))
    wanted=['FAP.CPU_FFT','FAP.frequency_transform_inclusive','FAP.CPU_iFFT','FAP.entire_augmentation_inclusive','ICS.extra_association_and_loss_inclusive','ICS.center_update','ICS.covariance','ICS.eigendecomposition','ICS.projection','APR.collect_filter_CPU','APR.history_transfer','APR.similarity_topk_loss']
    for r in read(RUN/'results/exp4/module_time_breakdown.csv'):
        if r['history_domains']=='3' and r['scope'].replace('DAMOT::','') in wanted:
            text.append('%s：CPU total=%s ms/step，CPU self=%s ms/step，GPU total=%s ms/step；嵌套total有重叠，不能相加。'%(r['scope'],r['CPU_total_ms_per_step'],r['CPU_self_ms_per_step'],r['GPU_total_ms_per_step']))
    text.append('APR库为真实帧、阶段模型与原筛选规则重建，仅供性能测量；0/1/2/3历史域均测，0域APR loss不工作。存储按tensor.numel()*element_size实测，Python容器不计；GPU临时拷贝与CPU候选并发快照见prototype_runtime_storage.csv。FAP/ICS/APR不在部署前向执行。')
    text.append('\n环境与可复现性：代码/数据/权重均缓存本地，NAS历史只读；Checkpoint/数据/代码哈希、环境版本、实际配置、PID/命令/退出记录和错误日志均保留。恢复环境与原日志版本不同（见environment_notes.txt），不声称严格复现原训练。全量图像解码、视频/帧完整性与模型分支预检均通过。未进行源训练、完整适应、多种子、参数搜索或绘图，也未修改论文及baseline=33.00。\n未完成项：无。重新执行入口：scripts/reproduce.sh（按完成标记恢复，不重复成功前向）。')
    (RUN/'summary.txt').write_text('\n'.join(text)+'\n')
    p=RUN/'progress.json';o=json.loads(p.read_text());o['status']='results_compiled_pending_acceptance';o['experiments']={str(i):'complete' for i in range(1,4)};o['experiments']['4']='measured_and_compiled_pending_acceptance';p.write_text(json.dumps(o,indent=2))
if __name__=='__main__':
    arg=sys.argv[1] if len(sys.argv)>1 else 'all'
    if arg=='all':summary_all()
    else:globals()['exp'+arg]()
