"""Required paired contrasts; no tuning or retraining based on test results."""
import argparse
from common import *

METRICS=['HOTA','DetA','IDF1','AssA','IDSW']
def interval(d):
    d=np.asarray(d,dtype=float);assert len(d)>0 and np.isfinite(d).all()
    rs=np.random.RandomState(0);samples=rs.randint(0,len(d),size=(10000,len(d)))
    low,high=np.percentile(d[samples].mean(1),[2.5,97.5])
    return dict(videos=len(d),mean=float(d.mean()),ci_low=float(low),ci_high=float(high))

def analyze(batch):
    if batch==1:
        domains=DOMAINS;labels=['C0_b0','C0_INT_b025','C0_RAND_b025','C2_b0','C2_b025','C4_b0','C4_b025']
        pairs=[('C2_b025',b) for b in ['C0_b0','C2_b0','C0_INT_b025','C4_b025']]
    else:
        domains=['overcast','rainy','dawn_dusk','night'];labels=['INT_b0','INT_b025','HF_b0','HF_b01']
        pairs=[('INT_b025',b) for b in ['INT_b0','HF_b0']]
    out=ROOT/'analysis'/('batch%d'%batch);out.mkdir(parents=True,exist_ok=True)
    aggregates=[];videos={}
    for label in labels:
        for domain in domains:
            path=ROOT/'results'/label/domain
            assert (path/'metrics_complete.json').exists(),path
            aggregates.extend(rows(path/'metrics.csv'))
            videos[label,domain]={(r['video'],r['category']):r for r in rows(path/'per_video.csv') if r['test']=='True'}
    if batch==2:
        for label in labels:
            for cat in CLASSES+['OVERALL']:
                data=[r for r in aggregates if r['model']==label and r['category']==cat]
                assert len(data)==4
                aggregates.append(dict(model=label,domain='four_domain_mean',category=cat,**{k:float(np.mean([float(r[k]) for r in data])) for k in METRICS}))
    effects=[];diffs=[]
    for new,ref in pairs:
        combined=[]
        for domain in domains:
            a=videos[new,domain];b=videos[ref,domain];assert a.keys()==b.keys()
            for key in sorted(a):
                vid,cat=key;r=dict(model=new,reference=ref,domain=domain,video=vid,category=cat)
                for k in METRICS:r[k+'_delta']=float(a[key][k])-float(b[key][k])
                combined.append(r);diffs.append(r)
        for domain in domains+['pooled_test_videos']:
            for cat in ['OVERALL','car']:
                selected=[r for r in combined if r['category']==cat and (domain=='pooled_test_videos' or r['domain']==domain)]
                for k in METRICS:effects.append(dict(model=new,reference=ref,domain=domain,category=cat,metric=k,**interval([r[k+'_delta'] for r in selected])))
    csvout(out/'aggregate_metrics.csv',aggregates);csvout(out/'paired_effects.csv',effects);csvout(out/'video_differences.csv',diffs)
    gates=[]
    if batch==1:
        for domain in domains+['pooled_test_videos']:
            for cat in ['OVERALL','car']:
                tests=[next(r for r in effects if r['reference']==ref and r['domain']==domain and r['category']==cat and r['metric']=='AssA') for ref in ['C0_b0','C4_b025']]
                gates.append(dict(domain=domain,category=cat,both_AssA_CI_strictly_positive=all(r['ci_low']>0 for r in tests),
                                  C2_trim_vs_C0_ci_low=tests[0]['ci_low'],C2_trim_vs_C4_trim_ci_low=tests[1]['ci_low']))
        csvout(out/'decision_gate.csv',gates)
    dump(out/'complete.json',dict(status='complete',batch=batch,bootstrap_seed=0,repetitions=10000,
         test_split='Exact ics_advantage deterministic MD5 video split',single_training_seed=True,
         statistic='Equal-video paired mean; percentile pointwise95% CI; pooled videos, no multiplicity correction',
         aggregate='Original native nonlinear domain aggregate; B2 mean is unweighted domain mean'))
    report()

def report():
    lines=['# 推理时 ICS 收缩：实验结果','',
        '仅推理与离线评估，单个训练种子。关联嵌入在归一化前执行 `z − β(zE)Eᵀ`，直接矩阵运算，基底 QR 正交化；无最小 RoI 数门槛。开关默认关闭。',
        '', '训练种子为1790355234；推理复现种子为20261029；bootstrap seed=0。设置名中 b0、b025、b01 分别表示 β=0、0.25、0.1。',
        '', 'β=0 对每个模型、每个域的全部检测与跟踪输出逐帧验证，与原冻结预测逐位一致后才接受该任务结果。',
        '', '## 第一批：C 系列 Night epoch2','']
    out=ROOT/'analysis/batch1'
    if (out/'complete.json').exists():
        agg=rows(out/'aggregate_metrics.csv');eff=rows(out/'paired_effects.csv');gates=rows(out/'decision_gate.csv')
        lines+=['C2 使用自己的固定干预基底，C4 使用自己的固定随机基底；C0 分别使用这两个基底作推理对照。数值为百分点，IDSW 为计数。','']
        lines+=['| 域 | 设置 | HOTA | DetA | IDF1 | AssA | IDSW |','|---|---|---:|---:|---:|---:|---:|']
        for r in agg:
            if r['category']=='OVERALL':lines.append('| %s | %s | %s |'%(r['domain'],r['model'],' | '.join('%.4f'%float(r[k]) if k!='IDSW' else str(int(float(r[k]))) for k in METRICS)))
        lines+=['','配对差值只使用 `ics_advantage` 相同的 test 视频，按视频等权；10,000 次 bootstrap，seed=0，95% 百分位区间。它与上面的全域非线性聚合差值口径不同。','',
                '| 域 | 对比 | 视频数 | ΔHOTA [95% CI] | ΔDetA [95% CI] | ΔIDF1 [95% CI] | ΔAssA [95% CI] | ΔIDSW [95% CI] |','|---|---|---:|---:|---:|---:|---:|---:|']
        for domain in DOMAINS+['pooled_test_videos']:
            for ref in ['C0_b0','C2_b0','C0_INT_b025','C4_b025']:
                data=[next(r for r in eff if r['domain']==domain and r['reference']==ref and r['category']=='OVERALL' and r['metric']==k) for k in METRICS]
                lines.append('| %s | C2_b025 − %s | %s | %s |'%(domain,ref,data[0]['videos'],' | '.join('%.4f [%.4f, %.4f]'%(float(r['mean']),float(r['ci_low']),float(r['ci_high'])) for r in data)))
        lines+=['','用户指定判据：C2_b025 相对 C0_b0 和 C4_b025 的 AssA 区间下界都严格大于0。各域分别检验；合并视频作为补充，避免将某一口径的结果推广到所有域。','']
        for r in gates:
            if r['category']=='OVERALL':lines.append('- %s：%s。'%(r['domain'],'满足“推理时收缩有效”判据' if r['both_AssA_CI_strictly_positive']=='True' else '未满足“推理时收缩有效”判据'))
        lines+=['','完整类别聚合、car 及 OVERALL 的配对结果、每视频差值和判据在 `analysis/batch1/` CSV 中。','']
    else:lines+=['正在进行；尚无第一批结论。','']
    lines+=['## 第二批：完整训练 stage4 epoch5 与 HF','']
    out=ROOT/'analysis/batch2'
    if (out/'complete.json').exists():
        lines+=['新完整运行 β∈{0,0.25}；HF β∈{0,0.1}；均使用各自 checkpoint 中实际保存的 bank_pcs 行向量，经 QR 后收缩。','',
                '| 域 | 设置 | HOTA | DetA | IDF1 | AssA | IDSW |','|---|---|---:|---:|---:|---:|---:|']
        for r in rows(out/'aggregate_metrics.csv'):
            if r['category']=='OVERALL':lines.append('| %s | %s | %s |'%(r['domain'],r['model'],' | '.join('%.4f'%float(r[k]) for k in METRICS)))
        effects=rows(out/'paired_effects.csv')
        lines+=['','| 域 | 对比 | 视频数 | ΔHOTA [95% CI] | ΔDetA [95% CI] | ΔIDF1 [95% CI] | ΔAssA [95% CI] | ΔIDSW [95% CI] |','|---|---|---:|---:|---:|---:|---:|---:|']
        for domain in ['overcast','rainy','dawn_dusk','night','pooled_test_videos']:
            for ref in ['INT_b0','HF_b0']:
                data=[next(r for r in effects if r['domain']==domain and r['reference']==ref and r['category']=='OVERALL' and r['metric']==k) for k in METRICS]
                lines.append('| %s | INT_b025 − %s | %s | %s |'%(domain,ref,data[0]['videos'],' | '.join('%.4f [%.4f, %.4f]'%(float(r['mean']),float(r['ci_low']),float(r['ci_high'])) for r in data)))
        lines+=['','按 OVERALL AssA 的配对区间判读（单种子、逐项区间）：','']
        for r in effects:
            if r['category']=='OVERALL' and r['metric']=='AssA':
                conclusion='区间完全为正' if float(r['ci_low'])>0 else '区间完全为负' if float(r['ci_high'])<0 else '区间包含0'
                lines.append('- %s，INT_b025 − %s：%s。'%(r['domain'],r['reference'],conclusion))
        lines+=['','四域的均值按域等权；pooled_test_videos 的配对统计按视频等权。两者口径不同。全部类别及 car 结果在 `analysis/batch2/`。','']
    else:
        status=ROOT/'manifests/batch2_status.json';s=read_json(status) if status.exists() else {'status':'waiting_for_full_training'}
        lines+=['尚未完成；等待独立完整训练及其冻结评估，当前状态：`%s`。'%s['status'],'']
        if s.get('stop'):
            d=s['stop']['decision']
            lines+=['源运行在 stage1 的注册停止指标 `bank PC1 / offline FAP PC1` 为 %.4f°，超过45°门槛；Stage4 checkpoint 尚不存在。'%d['first_direction_angle_deg'],
                    '', '源任务当前状态：`%s`。同时记录的第一方向到离线二维子空间夹角为 %.4f°，两个子空间主角为 %s；这些角度的定义不同。'%(s.get('source_status','unknown'),d['first_direction_to_offline_span_deg'],d['principal_angles_deg']),
                    '']
        context=ROOT/'manifests/source_readiness_context.json'
        if context.exists():
            current=read_json(context);gate=current['current_stage1_gate']
            if gate.get('user_authorized_interpretation_correction'):
                lines+=['源任务状态更新（%s）：门槛按其记录的授权解释修正为“第一方向到离线二维子空间”，夹角 %.4f°，判定 `%s`；当前控制器状态 `%s`。'%
                        (current['captured_at'],gate['gate_angle_deg'],gate['status'],current['formal']['status']),'']
        lines+=['按用户2026-10-31的明确选择，第二批保留 stage4 四域的原范围，等待源任务完成；stage1 结果只属于另一个任务的补充，不代替本任务第二批。','']
    lines+=['## 实现与可复现信息','',
            '- 源码为隔离副本；不调用 `bank.trim()`；单 RoI 的矩阵收缩及默认关闭路径自检通过。',
            '- 同帧同模型的不同 β 复用检测器前向，跟踪器状态彼此独立；β=0 的整域逐位一致检查保留在 `jobs/*/*/inference_complete.json`。',
            '- 训练 bank 的逐行更新在基向量非正交时与 QR 投影不完全相同；本实验按指定的 QR 子空间矩阵算子实现，第二批会记录基底及算子差异。',
            '- 大文件、预测和日志保存在 NAS；本地仅保留小型代码与短路径临时文件。',
            '- 全部结果为单种子；视频区间反映视频抽样变动，不代表训练种子不确定性。','']
    (ROOT/'SUMMARY.md').write_text('\n'.join(lines))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--batch',type=int,choices=[1,2]);p.add_argument('--report',action='store_true');a=p.parse_args()
    report() if a.report else analyze(a.batch)
