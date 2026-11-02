"""Write the complete HF-only report from the delivered numerical outputs."""
from common import *
import platform
import scipy
import matplotlib
import mkl

ORDER = ['Intervention-sensitive', 'Label-free nuisance', 'Random', 'Identity oracle']


def main():
    assert read_json(ROOT / 'analysis_complete.json')['status'] == 'PASS'
    rows = read_csv(ROOT / 'drift_summary.csv')
    full = read_csv(ROOT / 'drift_restoration_HF_native.csv')
    checks = read_csv(ROOT / 'restoration_selfchecks.csv')
    energies = read_csv(ROOT / 'S1_energy_check.csv')
    split = read_json(ROOT / 'split.json')
    review = read_json(ROOT / 'visual_review.json')
    assert review['status'] == 'reviewed'
    assert len(rows) == 12 and len(checks) == 13 and all(r['passed'] == 'True' for r in checks)
    one = {(r['basis'], int(r['k'])): r for r in rows}
    none = float(rows[0]['none_pp'])
    lines = ['# HF 原生 ICS 图表 v3', '',
        '纯 CPU 离线重算；只使用 HF 模型，不训练、不抽取新特征。全部新增文件在本目录。', '',
        '## 1. 输入与修正范围', '',
        '已执行 `git pull --ff-only origin main`，起点为 `%s`。' % BASE_COMMIT, '',
        'HF 特征根目录从 [idapr/inputs.txt](../idapr/inputs.txt) 的 `HF_feat_root` 读取：`%s`。'
        'HF_stage1_e5 五域的 `feat_clean` 和 `feat_fap` 均存在、维数为 256、数值有限；'
        '类别均为 car。各 NPZ 的路径、形状和 SHA-256 见 [input_manifest.json](input_manifest.json)。' % hf_root(), '',
        '旧 [idapr_offline.py](../idapr/source/tools/analysis/idapr_offline.py) 的 `run_model` '
        '对 F、HF 共用 `F_stage1_e5` 来估计 FAP residual PCA 与 Identity oracle。'
        '本版复用同一 `oracle_bases` 函数，将 reference 改为 **HF_stage1_e5**，只发布它的 FAP 与身份基底。'
        '这两个基底只在 HF stage1 fit 视频上估计；推理/训练实现未改动，也没有读取 F 特征。', '',
        'Label-free nuisance 沿用 A2 中已有的 **HF stage1** 行；Random 沿用 HF 的全部十个预定 replicate。'
        '任务 3 单独使用 HF stage4 的 INT 与无标签身份基底，不能把它们与 stage1 漂移基底混称。', '',
        'fit/test 视频划分与原 HF 完全一致：fit %d 个 RoI、%d 个视频，test %d 个 RoI、%d 个视频；'
        '视频不交叉。检索保持 gap=4、window=30、max_q=4000、seed=0，'
        'query_stage=4，gallery_stage=1/2/3，匹配旧域 Overcast/Rainy/Dawn-Dusk。' %
        (split['fit_rois'], len(split['fit_videos']), split['test_rois'], len(split['test_videos'])), '',
        '## 2. 漂移还原结果', '',
        '`恢复比例 (%) = 100 × Σ forgetting_reduction / Σ forgetting_proxy`，对三个旧域求和。', '',
        '`剩余遗忘 (pp) = 100 × Σ forgetting_restored / 3`。'
        '**未还原遗忘为 %.9f pp**，通过 1.798 pp 的自检。'
        '这里的遗忘是跨阶段身份检索 mAP 损失的代理，不能当作跟踪 AvgF。' % none, '',
        '| k | Intervention-sensitive (%) | Label-free nuisance (%) | Random mean ± SD (%) | Identity oracle (%) |',
        '|---:|---:|---:|---:|---:|']
    for k in KS:
        random = one[('Random', k)]
        lines.append('| %d | %.3f | %.3f | %.3f ± %.3f | %.3f |' %
                     (k, float(one[('Intervention-sensitive', k)]['recovered_pct']),
                      float(one[('Label-free nuisance', k)]['recovered_pct']),
                      float(random['recovered_pct']), float(random['random_sd']),
                      float(one[('Identity oracle', k)]['recovered_pct'])))
    lines += ['', '随机 SD 使用十个 replicate 各自的 ratio of sums，再计算总体 SD（ddof=0）；'
              '它描述基底间变动，不是训练种子置信区间。', '',
              '| k | 基底 | 恢复比例 (%) | 随机 SD | 随机 min/max (%) | 剩余遗忘 (pp) |',
              '|---:|---|---:|---:|---|---:|']
    for row in rows:
        random = row['basis'] == 'Random'
        lines.append('| %s | %s | %.6f | %s | %s | %.9f |' %
                     (row['k'], row['basis'], float(row['recovered_pct']),
                      '%.6f' % float(row['random_sd']) if random else '—',
                      '[%.6f, %.6f]' % (float(row['random_min']), float(row['random_max'])) if random else '—',
                      float(row['residual_pp'])))
    lines += ['', '逐域恢复比例如下。汇总列使用上面的 ratio of sums，并非下面三个百分比的平均。', '',
              '| k | 基底 | Overcast (%) | Rainy (%) | Dawn/Dusk (%) |',
              '|---:|---|---:|---:|---:|']
    for k in KS:
        for basis in ORDER:
            rr = [next(r for r in full if r['row_type'] == 'domain' and int(r['k']) == k
                       and r['basis'] == basis and r['domain'] == domain) for domain in TARGETS[:3]]
            lines.append('| %d | %s | %.6f | %.6f | %.6f |' %
                         (k, basis, *[float(r['recovered_pct']) for r in rr]))
    lines += ['', '完整的原始 mAP、逐域遗忘、汇总、逐域随机 replicate、随机汇总和 SD 均在 '
              '[drift_restoration_HF_native.csv](drift_restoration_HF_native.csv)（168 行数据）；'
              '其中 summary 行的三个原始 forgetting 字段是三域之和，mAP 字段留空；'
              'domain 行为该域本身。所有 mAP 原始字段为 0–1，`*_pct` 为百分数，`*_pp` 为百分点。', '',
              '### 与旧 F 参考基底结果的差别', '',
              '两版均使用相同 ratio-of-sums 口径，唯一区别是估计基底的参考模型。', '',
              '| k | 基底 | 旧 F_stage1 参考 (%) | 本版 HF_stage1 参考 (%) | 差值 (pp) |',
              '|---:|---|---:|---:|---:|']
    original = read_csv(IDAPR / 'A/A2_restoration.csv')
    for k in KS:
        for basis, old_name in [('Intervention-sensitive', 'FAP residual PCA'), ('Identity oracle', 'Identity oracle')]:
            rr = [r for r in original if r['model'] == 'HF' and r['query_stage'] == '4'
                  and int(r['k']) == k and r['basis'] == old_name]
            assert len(rr) == 3
            old = 100 * sum(float(r['forgetting_reduction']) for r in rr) / sum(float(r['forgetting_proxy']) for r in rr)
            new = float(one[(basis, k)]['recovered_pct'])
            lines.append('| %d | %s | %.6f | %.6f | %+.6f |' % (k, basis, old, new, new-old))
    lines += ['', 'HF 原生 k=2 干预方向恢复 %.3f%%，随机为 %.3f ± %.3f%%，'
              '身份 oracle 为 %.3f%%。这里只能描述干预前两维恢复很少、身份方向恢复更多；'
              '没有据此做随机基底间的显著性或等效性检验。'
              '扩展到 k=8/16 后干预方向恢复 %.3f%% / %.3f%%，不能把高维干预空间也说成不承载遗忘。' %
              (float(one[('Intervention-sensitive', 2)]['recovered_pct']),
               float(one[('Random', 2)]['recovered_pct']), float(one[('Random', 2)]['random_sd']),
               float(one[('Identity oracle', 2)]['recovered_pct']),
               float(one[('Intervention-sensitive', 8)]['recovered_pct']),
               float(one[('Intervention-sensitive', 16)]['recovered_pct'])), '',
              '### 自检', '', '| 检查 | 域 | k | 实际 | 参考 | 绝对误差 | 结果 |',
              '|---|---|---:|---:|---:|---:|---|']
    for r in checks:
        lines.append('| %s | %s | %s | %.12g | %.12g | %.3g | PASS |' %
                     (r['check'], r['domain'], r['k'], float(r['actual']), float(r['expected']), float(r['error'])))
    lines += ['', '**13/13 项通过**。九项逐域 baseline 与原 A2 对齐，最大允许误差 1e-9；'
              '随机均值与 1.2 / 5.7 / 12.5 的误差均小于 0.1；未还原遗忘误差小于 0.001 pp。', '',
              '![HF 四根柱](drift_restoration_HF_4bars.png)', '',
              '![HF 三根柱](drift_restoration_HF_3bars.png)', '',
              '两图均为单个 HF 子图，宽 3.3 英寸，字体至少 8pt，PNG 300dpi，PDF 同尺寸；'
              'k=2 组只加浅灰底，不加文字。图例置底。', '',
              '## 3. 固定二十个 RoI 截图', '',
              '严格复用 [plot_C_crops.csv](../ics_evidence/part1/HF/plot_C_crops.csv) 的二十行、'
              '顺序和浮点 bbox，布局仍为 2×10，不重新选点。去掉总标题；每个小图只有天气和投影坐标。'
              '原图解析路径、crop 尺寸、标签及文件哈希见 [crop_manifest.csv](crop_manifest.csv) 和 '
              '[crop_protocol.json](crop_protocol.json)。', '',
              '![HF INT extrema](intervention_extremes_HF.png)', '',
              '## 4. 表 S1：直接核对保存的 INT_00', '',
              '测试数据为 HF stage4 的全部 18,019 个 car RoI。真实跨天气偏移按 A1 定义：'
              '四个目标天气均值相对 Clear 的差，乘 sqrt(min(n_target,n_clear))；'
              '身份能量使用未修改的 `p.identity_rows`（视频内身份均值差，至少三次观测）。'
              '两项均计算 `sum((rows @ E)^2) / (sum(rows^2)+EPS)`，不是逐样本比例的平均。', '',
              '无标签身份基底按 A1 的 `label_free_bases` 在 HF stage4 fit car RoI 上重新估计，'
              '视图为 FAP/photo/student，eps_ratio=0.1；只向估计器提供图像成员关系和配对特征。'
              '它与任务 1 的“有身份标签的 stage1 oracle”不同。', '',
              '| 基底 | 跨天气能量 (%) | 身份能量 (%) | 对应 A1 | 天气差值 (pp) | 身份差值 (pp) | 与 A1 nuisance 的两个主角度 (°) |',
              '|---|---:|---:|---|---:|---:|---|']
    for r in energies:
        aa = json.loads(r['A1_nuisance_principal_angles_deg'])
        lines.append('| %s | %.9f | %.9f | HF/4 %s k=2 | %.3g | %.3g | %.12g, %.12g |' %
                     (r['basis'], float(r['cross_weather_pct']), float(r['identity_pct']), r['A1_kind'],
                      float(r['cross_weather_difference_pp']), float(r['identity_difference_pp']), aa[0], aa[1]))
    primary = [r for r in energies if r['basis'] in ['INT_00', 'Label-free identity']]
    matches = all(r['matches_prompt_rounding'] == 'True' for r in primary)
    int_result = next(r for r in primary if r['basis'] == 'INT_00')
    same_int = max(abs(float(int_result[k])) for k in ['cross_weather_difference_pp', 'identity_difference_pp']) < 1e-8
    same_int = same_int and float(int_result['A1_nuisance_angle_max_deg']) < 1e-5
    lines += ['', '**%s**：直接使用 INT_00 的结果%s 39.39%% / 11.11%%；'
              '无标签身份基底%s 2.55%% / 13.34%%。完整精度和每项差值见 '
              '[S1_energy_check.csv](S1_energy_check.csv)。所有解释以本次直接测量的 INT_00 为准。' %
              ('确认一致' if matches else '发现差异', '等于（按两位小数）' if matches else '不同于',
               '等于（按两位小数）' if matches else '需与原表逐项比较'), '',
              ('INT 与 A1 nuisance 本次在数值精度内相同；约 1e-14° 的角度和约 1e-14 pp 的差值属于数值精度。'
               '表 S1 可以引用该 INT 直接测量值。' if same_int else
               'INT 与 A1 nuisance 存在差异；表 S1 应改用上面直接测量的 INT_00 数字，并保留差异说明。'), '',
              '## 5. 复现与校验', '',
              '命令与依赖见 [README.txt](README.txt)，完整顺序见 [scripts/run_all.sh](scripts/run_all.sh)。'
              '运行日志在 [logs/](logs/)。输入/源码哈希在 [input_manifest.json](input_manifest.json)，'
              '检索任务哈希在 [retrieval_protocol.json](retrieval_protocol.json)，'
              '全包校验结果在 [validation.json](validation.json)。', '',
              '所有命令使用 OMP_NUM_THREADS=4、OPENBLAS_NUM_THREADS=4、MKL_NUM_THREADS=4、'
              'CUDA_VISIBLE_DEVICES 为空；没有临时降低 MKL 线程数。仅导入历史分析模块，不修改历史目录。', '']
    (ROOT / 'SUMMARY.md').write_text('\n'.join(lines).rstrip() + '\n')
    assert mkl.get_max_threads() == 4
    write_json(ROOT / 'runtime_versions.json', dict(python=platform.python_version(), numpy=np.__version__,
               scipy=scipy.__version__, matplotlib=matplotlib.__version__, executable=sys.executable,
               MKL_runtime_threads=mkl.get_max_threads(),
               environment={name: os.environ[name] for name in ['OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'CUDA_VISIBLE_DEVICES']}))
    print('REPORT_COMPLETE', len(rows), 'summary rows;', len(checks), 'checks; S1 matches:', matches, flush=True)


if __name__ == '__main__':
    main()
