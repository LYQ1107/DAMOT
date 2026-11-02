"""Combine the completed model tables and apply the prespecified P1--P4 rules."""
from common import *
import numpy as np
import platform
import sklearn
import scipy
import matplotlib

OUT = ROOT / 'quadrant'
SPACES = ['INT', 'Random', 'ID', 'W', 'ICS', 'Complement', 'Full']
LABELS = {'INT': 'INT 2D', 'Random': 'Random 2D (mean ± SD, n=10)', 'ID': 'ID oracle 2D',
          'W': 'W oracle 2D', 'ICS': 'ICS bank 2D', 'Complement': 'INT 补空间 254D', 'Full': 'Full 256D'}
METRICS = ['id_mAP_euc', 'id_mAP_cos', 'silhouette', 'within_var_ratio', 'weather_bacc', 'nightclear_bacc']


def metrics_table(rows, model):
    lines = ['| 空间 | ID mAP 欧氏 (%) | ID mAP 余弦 (%) | 身份轮廓系数 | 身份内部方差占比 ↓ | 天气 bAcc (%) | Night/Clear bAcc (%) |',
             '|---|---:|---:|---:|---:|---:|---:|']
    for space in SPACES:
        rr = [r for r in rows if r['model'] == model and r['space'] == space]
        assert len(rr) == (10 if space == 'Random' else 1)
        cells = [LABELS[space]]
        for metric in METRICS:
            if rr[0][metric] == '':
                cells.append('—')
                continue
            values = np.array([float(r[metric]) for r in rr])
            digits = 4 if metric in ['silhouette', 'within_var_ratio'] else 3
            fmt = '%%.%df' % digits
            value = fmt % values.mean()
            if space == 'Random':
                value += ' ± ' + fmt % values.std(ddof=0)
            cells.append(value)
        lines.append('| ' + ' | '.join(cells) + ' |')
    return lines


def main():
    rows = []
    for model in ['F', 'HF']:
        complete = json.loads((OUT / ('complete_%s.json' % model)).read_text())
        assert complete['status'] == 'complete' and complete['phase'] in ['all', 'figures']
        rr = read_csv(OUT / ('quadrant_metrics_%s.csv' % model))
        assert len(rr) == 16
        rows.extend(rr)
    write_csv(OUT / 'quadrant_metrics.csv', rows)
    tables = ['# Fixed-basis quadrant metrics', '',
              'mAP / bAcc 为百分数；轮廓系数和方差比为原值。Random SD 使用 ddof=0。', '']
    for model in ['F', 'HF']:
        tables += ['## ' + model, ''] + metrics_table(rows, model) + ['']
        ref = next(r for r in rows if r['model'] == model)
        tables += ['全部测试 RoI：%s；测试视频：%s。轮廓系数实际使用 %s 个 RoI、%s 个视频；'
                   '方差占比使用 %s 个 RoI、%s 个非恒定视频。检索沿用固定 %s 个 query，'
                   '使用完整测试 RoI 构建对应 gallery。' %
                   (ref['n_rois'], ref['n_test_videos'], ref['n_silhouette_rois'], ref['n_videos_used'],
                    ref['n_variance_rois'], ref['n_variance_videos'], ref['n_retrieval_queries']), '']
    (OUT / 'quadrant_metrics.md').write_text('\n'.join(tables).rstrip() + '\n')

    drift = read_csv(ROOT / 'drift/drift_restoration_summary.csv')
    checks = read_csv(ROOT / 'drift/selfchecks.csv')
    assert all(r['pass_check'] == 'True' for r in checks)
    review_path = ROOT / 'visual_review.json'
    review = json.loads(review_path.read_text())
    assert review['status'] == 'reviewed'
    lines = ['# ICS figures v2：CPU 离线重画与补充指标', '',
             '模型：F、HF 的 stage4 特征。全部修改和产物位于本目录；不训练、不做 GPU 推理。', '',
             '## 1. 漂移还原：统一使用 ratio of sums', '',
             '`恢复比例 = 100 × 三旧域 forgetting_reduction 之和 / 三旧域 forgetting_proxy 之和`。', '',
             '`剩余遗忘 = 100 × 三旧域 forgetting_restored 之和 / 3`。'
             '不再使用各域恢复比例的平均。这里的遗忘是跨阶段身份检索 mAP 损失的代理指标，'
             '不是实测跟踪 AvgF。', '',
             '**自检全部 PASS：%d/%d 项。** 恢复比例最大绝对误差 %.6f，低于要求 0.1。'
             'F 未还原损失 %.6f pp；HF %.6f pp。' %
             (len(checks), len(checks), max(float(r['absolute_error']) for r in checks if r['metric'] == 'recovered_pct'),
              float(next(r for r in drift if r['model'] == 'F')['none_pp']),
              float(next(r for r in drift if r['model'] == 'HF')['none_pp'])), '',
             '| 模型 | 基底 | k | 恢复比例 (%) | 随机 SD | 随机 min/max | 剩余遗忘 (pp) | 未还原 (pp) |',
             '|---|---|---:|---:|---:|---|---:|---:|']
    for r in drift:
        random = r['basis'] == 'Random'
        lines.append('| %s | %s | %s | %.3f | %s | %s | %.6f | %.6f |' %
                     (r['model'], r['basis'], r['k'], float(r['recovered_pct']),
                      '%.3f' % float(r['random_sd']) if random else '—',
                      '[%.3f, %.3f]' % (float(r['random_min']), float(r['random_max'])) if random else '—',
                      float(r['residual_pp']), float(r['none_pp'])))
    lines += ['', '随机对照先对每个预定 replicate 计算上述 ratio of sums，再报告十个值的均值、'
              '总体 SD（ddof=0）和 min/max。它们是随机基底间变动，不是训练种子或置信区间。', '',
              '来源：[A2_restoration.csv](../idapr/A/A2_restoration.csv)、'
              '[A2_random_replicates.csv](../idapr/A/A2_random_replicates.csv)。'
              '旧 [idAPR SUMMARY](../idapr/SUMMARY.md) 使用各域比例的平均，'
              '所以恢复百分比与本版不同；本版保留原 CSV，按用户指定新公式重算。', '',
              '正文 A：[双栏](drift/drift_restoration_A_double.pdf) / [单栏](drift/drift_restoration_A_single.pdf)；'
              '补充 B：[双栏](drift/drift_restoration_B_double.pdf) / [单栏](drift/drift_restoration_B_single.pdf)；'
              '补充 C：[双栏](drift/drift_restoration_C_double.pdf) / [单栏](drift/drift_restoration_C_single.pdf)。', '',
              '![A 双栏](drift/drift_restoration_A_double.png)', '',
              '## 2. 全测试集的空间指标', '',
              '数据完全复用未修改的 `evidence_cpu.load(model)`；基底直接读取已存 `.npy`。'
              '不重新估计基底，不对投影后的欧氏特征做归一化。天气探针只在 fit 视频拟合，'
              '测试使用所有 test RoI；同一身份标签只用于检索、聚集度评价和已冻结的 ID oracle。', '',
              '轮廓系数：每视频仅保留出现 ≥3 次的身份，至少两个合格身份，使用全部合格 RoI 的'
              '精确 Euclidean silhouette，再按 RoI 数加权。方差比：每个非恒定测试视频的'
              '身份内 SSE / 总 SSE，按该视频全部 RoI 数加权；不对该项额外筛除短身份。', '']
    lines += tables[4:]
    lines += ['## 3. 原表对齐自检', '',
              '| 模型 | 空间 | 指标 | 新值 | 原表精确值 | 绝对差 | 结果 |',
              '|---|---|---|---:|---:|---:|---|']
    for model in ['F', 'HF']:
        for r in read_csv(OUT / ('selfchecks_%s.csv' % model)):
            assert r['pass_check'] == 'True'
            lines.append('| %s | %s | %s | %.9f | %.9f | %.12g | PASS |' %
                         (model, r['space'], r['metric'], float(r['actual']), float(r['reference_exact']),
                          float(r['absolute_error'])))
    lines += ['', '对应原表：[ics_evidence Table 2b](../ics_evidence/SUMMARY.md#table-2b--delete-each-2d-subspace-β1-④)。'
              'Full 对应 Full 行，补空间对应 INT 删除行；余弦 mAP 另与原表 1 clean 行一致。', '',
              '数值兼容说明：启动命令保持 OMP/OPENBLAS/MKL=4、CUDA_VISIBLE_DEVICES 为空。'
              '历史 [终端记录](../ics_evidence/logs/part1_F_terminal_transcript.txt) 明确使用 2 线程。'
              '未修改的 L-BFGS 天气探针在 4 线程下会改变少数预测；'
              '本版仅在历史加载/投影/分类探针兼容计算中临时使用 MKL=2，并恢复到 4；'
              'PCA、t-SNE、轮廓系数等保持 4 线程。'
              '诊断数据见 [probe_numerical_diagnostic.csv](logs/probe_numerical_diagnostic.csv)，'
              '首轮失败日志和 4 线程结果也完整保留。未放宽自检误差。', '',
              '## 4. 预先指定的 P1–P4', '']
    for model in ['F', 'HF']:
        d = json.loads((OUT / ('decisions_%s.json' % model)).read_text())
        lines += ['### ' + model, '',
                  '- **P1 %s**：INT 天气 bAcc %.6f；Random %.6f ± %.6f；'
                  '严格阈值 mean + 3SD = %.6f。' %
                  (d['P1']['status'], d['P1']['INT_weather_bacc'], d['P1']['random_mean'],
                   d['P1']['random_sd'], d['P1']['threshold']),
                  '- **P2 %s**：欧氏身份 mAP，INT %.6f、ID oracle %.6f；'
                  '轮廓系数，INT %.6f、ID oracle %.6f。要求两项均严格低于 ID oracle。' %
                  (d['P2']['status'], d['P2']['INT_id_mAP'], d['P2']['ID_id_mAP'],
                   d['P2']['INT_silhouette'], d['P2']['ID_silhouette'])]
        translations = {'higher': '高于', 'lower': '低于', 'comparable': '相当（在 ±2SD 内）'}
        for metric, r in d['P3'].items():
            lines.append('- **P3 %s：相对随机为“%s”**。INT %.6f；Random %.6f ± %.6f；'
                         '预定 ±2SD 区间 [%.6f, %.6f]。' %
                         (metric, translations[r['result']], r['INT'], r['random_mean'], r['random_sd'],
                          r['band_low'], r['band_high']))
        lines += ['- **P4 %s**：补空间余弦 mAP %.6f / Full %.6f = %.6f%%，要求 ≥98%%。' %
                  (d['P4']['status'], d['P4']['complement_cosine_mAP'], d['P4']['full_cosine_mAP'],
                   100 * d['P4']['retention_ratio']), '']
    lines += ['P3 的“相当”仅指用户预设的十个随机基底均值 ±2SD 描述性区间，'
              '不是正式等效性检验；P1 的 3SD 阈值也不解释为独立显著性检验。', '',
              '## 5. PCA 与 t-SNE 的图像观察', '']
    for observation in review['observations']:
        lines += [observation, '']
    for model in ['F', 'HF']:
        lines += ['### ' + model, '',
                  '![%s PCA](quadrant/quadrant_pca_%s.png)' % (model, model), '',
                  '![%s t-SNE](quadrant/quadrant_tsne_%s.png)' % (model, model), '']
    lines += ['图中所有列使用同一种降维方法；数值标注来自降维前的全测试集。'
              '每模型的两套图复用原 `quadrant_indices` 和 `highlighted_identities`，'
              'Random 固定 R_00；表格 Random 使用全部十个基底。'
              't-SNE 的簇形状不替代原空间的欧氏聚集度和检索量化指标。', '',
              '## 6. 复现与产物', '',
              '见 [README.txt](README.txt)、[scripts/run_all.sh](scripts/run_all.sh)。'
              '脚本仅 import 历史模块，未改动历史文件。'
              '原始特征和源码哈希见 `quadrant/metrics_protocol_{F,HF}.json`；'
              '图坐标 CSV、固定采样清单、全量逐基底指标和运行日志均随本目录交付。', '']
    (ROOT / 'SUMMARY.md').write_text('\n'.join(lines).rstrip() + '\n')
    write_json(ROOT / 'runtime_versions.json', dict(python=platform.python_version(), numpy=np.__version__,
        scipy=scipy.__version__, sklearn=sklearn.__version__, matplotlib=matplotlib.__version__,
        executable=sys.executable, shell_threads={k: os.environ[k] for k in
        ['OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'CUDA_VISIBLE_DEVICES']},
        random_sd_ddof=0, default_MKL_threads=4, historical_probe_MKL_threads=2))
    print('REPORT_COMPLETE', len(rows), 'metric rows', flush=True)


if __name__ == '__main__':
    main()
