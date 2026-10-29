#!/usr/bin/env python3
"""Package the completed BDD probe without the large raw feature arrays."""
from pathlib import Path
import csv
import datetime
import hashlib
import json
import tarfile

ROOT = Path(__file__).resolve().parent
ANALYSIS = ROOT / 'analysis_F_stage4_e5'
DOMAINS = ['clear_holdout_45v', 'overcast', 'rainy', 'dawn_dusk', 'night']
LABELS = ['F_stage%d_e5' % i for i in range(1, 5)]


def main():
    summary = json.loads((ANALYSIS / 'summary.json').read_text())
    p1 = {r['subspace']: r for r in summary['P1']}
    ics, random, chance = p1['ICS (checkpoint)'], p1['Random (mean)'], p1['Chance m/D']
    p2 = {r['space']: r for r in summary['P2']}
    protocol = summary['protocol']
    lines = [
        'DAMOT ICS 子空间探针：BDD 冻结检查点真实结果',
        '生成时间（北京时间）：' + datetime.datetime.now(datetime.timezone(datetime.timedelta(hours=8))).isoformat(),
        '',
        '已完成 BDD P1–P5；未训练 R2；未执行可选 VKITTI2 P6。',
        '主模型：F_stage4_e5。阶段：F_stage1–4_e5。D=256，m=2。',
        '模型权重冻结；仅离线天气线性分类器在 fit 视频上拟合。',
        '抽样：每域最多40视频，每视频前30帧、步长2、每帧最多40个car框，最小边16像素。',
        'Clear 使用独立45视频holdout中的40视频；四个目标域来自原适应池。',
        '计划抽样共148视频、4440帧；输出仅保留通过car/ignore/min-box筛选的RoI。',
        f"主模型共 {protocol['rois']} RoI；fit={protocol['fit_rois']}，test={protocol['test_rois']}。",
        '',
        'P1：测试视频上的投影能量占比',
        '类别内真实域均值相对Clear的偏移；BDD未配对，域差异不等价于纯天气变化。',
    ]
    for key in ['Real weather shift', 'FAP residual', 'Photometric residual',
                'Student-view residual', 'Identity', 'Temporal (same ID)']:
        if key in ics:
            lines.append('%s: ICS=%.4f%%; Random=%.4f%%; chance=%.4f%%; ICS/chance=%.3fx' %
                         (key, 100 * ics[key], 100 * random[key], 100 * chance[key],
                          ics[key] / chance[key]))
    lines += [
        'Weather/Identity=%.4f' % ics['Weather/Identity'],
        '天气能量是否达到预设10×chance阈值：' +
        ('达到' if ics['Real weather shift'] >= 10 * chance['Real weather shift'] else '未达到'),
        '', 'P2：按视频划分的天气线性探针与测试视频内身份检索',
    ]
    for name, row in p2.items():
        lines.append('%s: weather balanced accuracy=%.4f%%; ID mAP=%.4f%%' %
                     (name, 100 * row['weather_bacc'], 100 * row['id_mAP']))
    lines += ['检索查询数：%d；天气随机准确率：%.1f%%' %
              (summary['n_retrieval_queries'], 100 * next(iter(p2.values()))['chance']),
              '', 'P3：离线特征干预；每档重拟合天气分类器，未重新跑跟踪器/HOTA。',
              '下列差值均为百分点（pp）；不应把部分可逆缩放视为必然删除信息。']
    for name in ['ICS', 'Random', 'Identity oracle']:
        rows = {r['beta']: r for r in summary['P3'] if r['subspace'] == name}
        zero = rows[0.0]
        for beta in [.1, 1.0]:
            row = rows[beta]
            lines.append('%s beta=%.2f: weather %.4f%% (%+.4f pp), ID mAP %.4f%% (%+.4f pp), '
                         'domain-centroid gap %.6f -> %.6f' %
                         (name, beta, 100 * row['weather_bacc'],
                          100 * (row['weather_bacc'] - zero['weather_bacc']),
                          100 * row['id_mAP'], 100 * (row['id_mAP'] - zero['id_mAP']),
                          zero['weather_gap'], row['weather_gap']))
    lines += ['', 'P4：剂量响应（仅测试视频、默认每视频前三个抽样帧）']
    for family in sorted({r['family'] for r in summary['P4']}):
        rows = sorted([r for r in summary['P4'] if r['family'] == family], key=lambda r: r['strength'])
        monotonic = all(b['in_E_norm'] + 1e-10 >= a['in_E_norm'] for a, b in zip(rows, rows[1:]))
        lines.append('%s: 子空间内变化是否单调增加=%s；能量占比 %.4f%%–%.4f%%' %
                     (family, '是' if monotonic else '否',
                      100 * min(r['share_in_E'] for r in rows),
                      100 * max(r['share_in_E'] for r in rows)))
    lines += ['', 'P5：每阶段使用自身嵌入与ICS子空间，全部相同抽样RoI。']
    for row in summary['P5']:
        values = {k[6:]: v for k, v in row.items() if k.startswith('shift:')}
        peak = max(values, key=values.get)
        lines.append('After %s: peak=%s (%.4f%%); adapted-domain share=%.4f%%' %
                     (row['adapted_domain'], peak, 100 * values[peak],
                      100 * values[row['adapted_domain']] ))
    lines += ['', '验证与证据',
              '合成数据7项自检全部PASS；16视图的串行/多进程特征最大差值全部为0。',
              '全部20个(阶段,域)特征文件逐项校验；跨阶段video/ID/frame/image/category/bbox完全相同。',
              '每个检查点的参数/buffer匹配情况、ICS初始化与更新数见feats/*/*meta.json。',
              '图表可读性检查见readability_review.json；全部图见analysis_F_stage4_e5/figs/all_figures.pdf。',
              '表格、完整CSV和summary.json保留全部结果，不根据结论筛选样本或种子。',
              '', '解释范围',
              'BDD域差异同时包含天气、场景、相机和车辆组成变化；本实验不单独识别纯天气因果效应。',
              'student探针固定几何，仅保留颜色与FAP；训练的另一对比视图还包含翻转、裁剪等几何变化。',
              '低维探针不能证明任何身份信息绝对不存在；应结合能量、检索和负对照共同判断。',
              'R2配置已修正并验证为只更换估计方式，保留min_samples=96。训练仍需单独执行。',
              '原始npz留在服务器feats/供复画；本交付包只含meta，不重复传输npz。', '']
    (ROOT / 'RESULTS.txt').write_text('\n'.join(lines))
    figures = json.loads((ANALYSIS / 'figure_manifest.json').read_text())
    assert figures['page_count'] == 7, figures
    assert len(list((ANALYSIS / 'tables').glob('*.tex'))) == 3
    review = json.loads((ROOT / 'readability_review.json').read_text())
    assert review['status'] == 'passed'
    validation = dict(status='passed', figure_pages=7, latex_tables=3,
                      validated_feature_files=len(json.loads((ROOT / 'feature_validation.json').read_text())),
                      selftest_passes=(ROOT / 'logs/selftest_final_full.log').read_text().count('[PASS]'),
                      no_raw_npz_in_delivery=True)
    assert validation['validated_feature_files'] == 20 and validation['selftest_passes'] == 7
    (ROOT / 'delivery_validation.json').write_text(json.dumps(validation, indent=2))
    selected = []
    for directory in [ANALYSIS, ROOT / 'logs', ROOT / 'configs', ROOT / 'inputs']:
        selected.extend(p for p in directory.rglob('*') if p.is_file())
    selected.extend(ROOT.glob('*.json'))
    selected.extend(ROOT.glob('*.txt'))
    selected.extend(ROOT.glob('*.py'))
    selected.append(ROOT / 'ics_probe_corrected.patch')
    for label in LABELS:
        selected.extend((ROOT / 'feats' / label).glob('*meta.json'))
    selected = sorted({p for p in selected if p.name != 'delivery_file_manifest.json'})
    manifest = [dict(path=str(p.relative_to(ROOT)), bytes=p.stat().st_size,
                     sha256=hashlib.sha256(p.read_bytes()).hexdigest()) for p in selected]
    index = ROOT / 'delivery_file_manifest.json'
    index.write_text(json.dumps(manifest, indent=2))
    selected.append(index)
    archive = ROOT.parent / 'ics_probe_delivery.tar.gz'
    with tarfile.open(str(archive), 'w:gz') as stream:
        for p in selected:
            stream.add(str(p), arcname='ics_probe/' + str(p.relative_to(ROOT)), recursive=False)
    digest = hashlib.sha256(archive.read_bytes()).hexdigest()
    archive.with_name(archive.name + '.sha256').write_text(digest + '  ' + archive.name + '\n')
    print(json.dumps(dict(archive=str(archive), bytes=archive.stat().st_size,
                          files=len(selected), sha256=digest), indent=2))


if __name__ == '__main__':
    main()
