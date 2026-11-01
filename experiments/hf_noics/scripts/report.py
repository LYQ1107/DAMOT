"""Measured three-way single-seed comparisons and the fixed simplification decision."""
import csv
import json
from pathlib import Path
Q=Path(__file__).resolve().parents[1]
DOMAINS=['overcast','rainy','dawn_dusk','night']
METRICS=['HOTA','DetA','IDF1','AssA']
STATISTICS=['FinalMean','DiagonalMean','AIP','AvgF','FinalOld','BWT']
def read(path):
    with path.open() as f:return list(csv.DictReader(f))
def write(path,rows):
    if not rows:return
    path.parent.mkdir(parents=True,exist_ok=True)
    with path.open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
def statistics(rows,label):
    out=[]
    for category in ['OVERALL','car']:
        for metric in METRICS:
            c={(int(r['stage']),r['eval_domain']):float(r[metric+'_percent']) for r in rows if r['category']==category}
            assert len(c)==10
            diag=[c[j,DOMAINS[j-1]] for j in range(1,5)]
            out.append(dict(model=label,category=category,metric=metric,
                FinalMean=sum(c[4,d] for d in DOMAINS)/4,DiagonalMean=sum(diag)/4,
                AIP=sum(sum(c[i,d] for d in DOMAINS[:i])/i for i in range(1,5))/4,
                AvgF=sum(max(c[i,DOMAINS[j-1]] for i in range(j,4))-c[4,DOMAINS[j-1]] for j in range(1,4))/3,
                FinalOld=sum(c[4,d] for d in DOMAINS[:3])/3,
                BWT=sum(c[4,DOMAINS[j-1]]-diag[j-1] for j in range(1,4))/3))
    return out
rows=[]
for stage in range(1,5):
    for domain in DOMAINS[:stage]:
        c=Q/'results'/('stage%d'%stage)/domain
        if (c/'complete.json').exists():rows.extend(read(c/'metrics.csv'))
references={label:read(Q/'reference'/label/'stage_domain_metrics.csv') for label in ['HF','INTICS']}
full=len(rows)==70
write(Q/'results/stage_domain_metrics.csv',rows)
paired=[]
for label,reference in references.items():
    index={(int(r['stage']),r['eval_domain'],r['category']):r for r in reference}
    for r in rows:
        old=index[int(r['stage']),r['eval_domain'],r['category']]
        assert int(r['GT_count'])==int(old['GT_count'])
        v=dict(model='NoICS',reference=label,stage=int(r['stage']),eval_domain=r['eval_domain'],category=r['category'])
        for m in METRICS:
            v['NoICS_'+m]=float(r[m+'_percent']);v['reference_'+m]=float(old[m+'_percent']);v['delta_'+m]=v['NoICS_'+m]-v['reference_'+m]
        paired.append(v)
write(Q/'results/paired_stage_deltas.csv',paired)
stats=[]
for label,data in references.items():stats.extend(statistics(data,label))
if full:stats.extend(statistics(rows,'NoICS'))
write(Q/'results/continual_metrics.csv',stats)
delta=[]
if full:
    for new in [r for r in stats if r['model']=='NoICS']:
        for label in ['HF','INTICS']:
            old=next(r for r in stats if r['model']==label and r['category']==new['category'] and r['metric']==new['metric'])
            delta.append(dict(model='NoICS',reference=label,category=new['category'],metric=new['metric'],**{k:new[k]-old[k] for k in STATISTICS}))
    write(Q/'results/paired_continual_deltas.csv',delta)
hf=next(r for r in stats if r['model']=='HF' and r['category']=='OVERALL' and r['metric']=='HOTA')
decision=dict(status='pending',training_seed=1790355234,single_training_seed=True,category='OVERALL',
    criteria=dict(FinalMean_delta_min=-.3,AvgF_delta_max=.3,both_required=True),
    HF_FinalMean=hf['FinalMean'],HF_AvgF=hf['AvgF'],
    required_NoICS_FinalMean_min=hf['FinalMean']-.3,required_NoICS_AvgF_max=hf['AvgF']+.3)
if full:
    new=next(r for r in stats if r['model']=='NoICS' and r['category']=='OVERALL' and r['metric']=='HOTA')
    dh=new['FinalMean']-hf['FinalMean'];df=new['AvgF']-hf['AvgF']
    passed=dh>=-.3 and df<=.3
    decision.update(status='complete',NoICS_FinalMean=new['FinalMean'],NoICS_AvgF=new['AvgF'],FinalMean_delta=dh,AvgF_delta=df,
        HOTA_condition_passed=dh>=-.3,AvgF_condition_passed=df<=.3,equivalent_by_prespecified_rule=passed,
        main_method='FAP + APR; ICS and coupling supplementary' if passed else 'retain HF; report NoICS ablation')
(Q/'results/decision.json').write_text(json.dumps(decision,indent=2)+'\n')
lines=['# HF NoICS: FAP + APR','']
if full:
    lines+=['**预先固定判读：'+('简化版与 HF 相当；主方法采用 FAP + APR，ICS 与耦合移到补充材料。' if passed else '未达到与 HF 相当的条件；主方法保留 HF，并报告本次消融。')+'**',
        'OVERALL 最终四域平均 HOTA %.6f vs HF %.6f（Δ %+.6f，要求 ≥−0.3）；HOTA AvgF %.6f vs HF %.6f（Δ %+.6f，要求 ≤+0.3）。两项须同时满足，判定采用未舍入值。'%
        (new['FinalMean'],hf['FinalMean'],dh,new['AvgF'],hf['AvgF'],df),'']
else:lines+=['**预先固定判读：待四域结果完成后计算，当前不作结论。**',
    '要求 OVERALL 最终四域平均 HOTA ≥ %.6f，且 HOTA AvgF ≤ %.6f；两项同时满足才采用 FAP + APR。'%
    (decision['required_NoICS_FinalMean_min'],decision['required_NoICS_AvgF_max']),'']
if full:
    association={r['category']:r['AvgF'] for r in delta if r['reference']=='HF' and r['metric']=='AssA'}
    lines+=['关联遗忘的差异另行保留：NoICS−HF 的 AssA AvgF 为 OVERALL %+.6f、car %+.6f（增加表示遗忘更多）。主结论限于上述预先指定的 HOTA 判据和本次单种子配对。'%
        (association['OVERALL'],association['car']),'']
lines+=['单训练种子 **1790355234**，与 HF 配对。Clear S → Overcast → Rainy → Dawn/Dusk → Night，每域5 epoch，4张V100，FP32，16 samples/GPU，2 workers/GPU。几何、student低频FAP、contrastive高频噪声和APR历史规则保持HF取值。仅改变 `model.track_head.use_causal_direction_bank=False` 和 `roi_proto_couple_ics=False`。APR 使用原始全空间余弦分支；其余 causal_*、roi_proto_* 配置均保留。',
    '', '配置差异与输出路径、输入别名、只读审计hook分列于 [config_diff.json](config_diff.json)。训练从同一 Clear S 重新开始，冒烟权重不进入正式训练；四阶段没有停止门槛。','']
for name in ['V1_config','V2_smoke','V3_code_snapshot']:
    p=Q/'preflight'/(name+'.json')
    if not p.exists():continue
    v=json.loads(p.read_text())
    if name=='V1_config':lines.append('- V1：'+v['status']+'；每阶段模型部分恰好两个指定字段不同。')
    elif name=='V3_code_snapshot':lines.append('- V3：'+v['status']+'；442个快照文件（346个Python文件）逐文件SHA256相同；[比较清单](preflight/V3_code_snapshot.json)。同时与 INTICS 使用的原HF快照346个Python文件全部一致。HF初始清单中一个评估报表文件条目已过时，其后续更新记录与实际HF文件一致；训练文件无差异。')
    else:
        lines+=['- V2：'+v['status']+'；HF/NoICS 各50次更新，四阶段分配13/13/12/12，同一组四卡；APR历史域数0/1/2/3，stage1零、后续有限且非零，NoICS无bank模块和收缩损失。',
            '  每阶段前3次作为计时预热，余38次取每迭代最慢rank后平均：HF %.6fs，NoICS %.6fs，变化 %+.4f%%。全部逐迭代数据与训练计算时间见 [V2](preflight/V2_smoke.json)。'%
            (v['timings']['HF']['critical_rank_iteration_mean_steady38'],v['timings']['NoICS']['critical_rank_iteration_mean_steady38'],v['steady_iteration_time_change_percent'])]
lines+=['', '运行适配披露：HF源文件逐字节不改。外部启动器将状态日志改为兼容 `bank=None`，并将“bank必须初始化”和“耦合APR计数增加”两项断言替换为“bank不存在”和“原始全空间APR实际生效”的检查。训练阶段循环AST除此两项断言外保持相同；模型、损失、优化器、EMA、采样、学习率和阶段历史保存使用HF原实现。新增hook只读取原forward返回的损失对象并记录状态。见 [静态核对](preflight/runner_static_check.json) 与 [外部启动器](scripts/runtime_adapter.py)。HF原checkpoint元数据中的旧实验标签保留，实际设置以本配置与manifest为准。',
    '', '输出保存在本地任务目录；输入通过本任务的私有只读NAS挂载和原本地缓存别名访问相同图像。标注与全帧清单、代表图像哈希已核对，其他任务目录保持只读。阶段快照可在训练继续时用完全空闲的其他GPU评估，最多两个推理任务同时运行；复用单元核对checkpoint哈希。','',
    '冻结评估使用HF相同协议：detector .05、tracker .30/.20、Conv-BN fusion、all-frame HOTA。目前完成 %d/10 单元。'% (len(rows)//7),'',
    '| Stage | 域 | 类别 | HOTA | DetA | IDF1 | AssA |', '|---:|---|---|---:|---:|---:|---:|']
for r in rows:
    if r['category'] in ['OVERALL','car']:lines.append('| %s | %s | %s | %s |'%(r['stage'],r['eval_domain'],r['category'],' | '.join('%.6f'%float(r[m+'_percent']) for m in METRICS)))
lines+=['', '| Stage | 域 | 类别 | NoICS−HF ΔHOTA | NoICS−INTICS ΔHOTA |','|---:|---|---|---:|---:|']
for r in [r for r in paired if r['reference']=='HF' and r['category'] in ['OVERALL','car']]:
    other=next(x for x in paired if x['reference']=='INTICS' and (x['stage'],x['eval_domain'],x['category'])==(r['stage'],r['eval_domain'],r['category']))
    lines.append('| %s | %s | %s | %+.6f | %+.6f |'%(r['stage'],r['eval_domain'],r['category'],r['delta_HOTA'],other['delta_HOTA']))
if full:
    lines+=['', '连续学习口径与 INTICS 相同：FinalMean=四个R4j的平均；DiagonalMean=四个Rjj的平均；AIP=四阶段“当时已见域均值”的平均；AvgF=三个旧域 max_i=j..3(Rij)−R4j 的平均，不裁零；FinalOld=三个旧域R4j均值；BWT=三个旧域R4j−Rjj均值。AvgF越小越好，其余指标越大越好。','',
        '| 模型 | 类别 | 指标 | FinalMean | DiagonalMean | AIP | AvgF | FinalOld | BWT |','|---|---|---|---:|---:|---:|---:|---:|---:|']
    for r in stats:lines.append('| %s | %s | %s | %s |'%(r['model'],r['category'],r['metric'],' | '.join('%.6f'%r[k] for k in STATISTICS)))
    lines+=['', '| 比较 | 类别 | 指标 | ΔFinalMean | ΔDiagonalMean | ΔAIP | ΔAvgF | ΔFinalOld | ΔBWT |','|---|---|---|---:|---:|---:|---:|---:|---:|']
    for r in delta:lines.append('| NoICS−%s | %s | %s | %s |'%(r['reference'],r['category'],r['metric'],' | '.join('%+.6f'%r[k] for k in STATISTICS)))
lines+=['', '完整精度：[各阶段×域](results/stage_domain_metrics.csv)、[配对差值](results/paired_stage_deltas.csv)、[连续学习指标](results/continual_metrics.csv)。','']
vp=Q/'results/video_pairs/complete.json'
if vp.exists():
    video_meta=json.loads(vp.read_text());effects=read(Q/'results/video_pairs/paired_summary.csv')
    lines+=['逐视频比较严格复用 `ics_advantage`：视频名MD5、split seed0，取test视频；HOTA/AssA来自原TrackEval详细数组，IDF1来自原MMTracking/MOTMetrics逐视频累积。每组先核对官方聚合一致性。每个test视频等权；10000次有放回视频bootstrap、seed0、平均差值的pointwise percentile95% CI，中位数另报。区间反映视频抽样差异，仍只有一个训练种子；不参与上述预固定主判读。',
        '各域test视频数：'+json.dumps(video_meta['test_videos_by_domain'])+'。',
        '', '| 域 | 类别 | 指标 | 视频数 | 平均差值 | 中位数 | 平均差值95% CI |','|---|---|---|---:|---:|---:|---|']
    for r in effects:
        lines.append('| %s | %s | %s | %s | %+.6f | %+.6f | [%+.6f, %+.6f] |'%(r['domain'],r['category'],r['metric'],r['videos'],float(r['mean']),float(r['median']),float(r['ci_low']),float(r['ci_high'])))
    lines+=['', '四域pool行按视频等权，区别于主比较的四域宏平均和非线性官方聚合。原始配对：[逐视频差值](results/video_pairs/video_differences.csv)；[统计表](results/video_pairs/paired_summary.csv)；[来源与复用核对](manifests/video_statistics_provenance.json)。','']
else:lines+=['逐视频NoICS−HF bootstrap尚未完成，完成后补入；缺失值不作零处理。','']
optional=Q/'optional_drift/complete.json'
if optional.exists():
    lines+=['可选S3漂移分解已完成，与历史F、HF并列。共同参考基底为F stage1的fit视频，统计用test视频；模型间沿用历史共享坐标约定，未作额外对齐，也不主张坐标规范不变性。NoICS没有bank，因此ICS bank/covariance专属行不适用并省略，未估计替代bank。',
        '', 'Stage3→4、旧域test RoI的固定参考子空间漂移：',
        '', '| 模型 | 基底 | 请求维数 | 实际维数 | 平均能量占比 | 子空间内均方漂移 | 子空间外均方漂移 |',
        '|---|---|---:|---:|---:|---:|---:|']
    for r in read(Q/'optional_drift/drift_energy.csv'):
        if r['from_stage']=='3' and r['to_stage']=='4' and r['bucket']=='old' and not r['basis'].startswith('ICS'):
            lines.append('| %s | %s | %s | %s | %.6f%% | %.8f | %.8f |'%(r['model'],r['basis'],r['k_requested'],r['k_effective'],100*float(r['mean_energy_share']),float(r['mean_in_sq']),float(r['mean_out_sq'])))
    lines+=['', '全转换与检索恢复：[漂移分解](optional_drift/drift_energy.csv)、[跨阶段检索](optional_drift/cross_stage_retrieval.csv)、[漂移恢复](optional_drift/drift_restoration.csv)、[NoICS−HF/F差值](optional_drift/paired_drift_comparison.csv)。','']
else:
    p=Q/'manifests/optional_controller.json'
    phase=json.loads(p.read_text())['status'] if p.exists() else 'not started'
    lines+=['可选S3漂移分解状态：'+phase+'；不影响主判读和必需交付。','']
(Q/'SUMMARY.md').write_text('\n'.join(lines))
(Q/'manifests/result_status.json').write_text(json.dumps(dict(status='complete' if full and vp.exists() else 'in_progress',metric_rows=len(rows),evaluation_cells=len(rows)//7,decision_status=decision['status'],single_training_seed=True,seed=1790355234),indent=2)+'\n')
print('REPORT_READY',len(rows),decision['status'])
