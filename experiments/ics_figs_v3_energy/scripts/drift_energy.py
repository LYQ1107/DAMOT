"""HF drift-energy ratios using immutable v3/A2 bases; CPU only."""
import csv
import hashlib
import json
import os
import platform
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
REPO = ROOT.parents[1]
V3 = REPO / 'experiments/ics_figs_v3'
IDAPR = REPO / 'experiments/idapr'
BASE_COMMIT = '68595ecad85a987008cb53523ccc401d844b10fd'
KS = [2, 8, 16]
BASIS_ORDER = ['Intervention-sensitive', 'Random', 'Identity oracle']

assert os.environ.get('CUDA_VISIBLE_DEVICES') == '', 'CPU-only run required'
for key in ['OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS']:
    assert os.environ.get(key) == '4', (key, os.environ.get(key))
sys.dont_write_bytecode = True
sys.path[:0] = [str(REPO / 'experiments/ics_evidence/scripts'), str(REPO / 'tools/analysis')]

import numpy as np
import mkl
import ics_probe_analyze as p
import ics_probe_drift as drift


def log(message):
    print(time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()), message, flush=True)


def read_csv(path):
    with Path(path).open(newline='') as handle:
        return list(csv.DictReader(handle))


def write_csv(path, rows):
    fields = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with Path(path).open('w', newline='') as handle:
        writer = csv.DictWriter(handle, fields, lineterminator='\n')
        writer.writeheader()
        writer.writerows(rows)


def read_json(path):
    return json.loads(Path(path).read_text())


def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + '\n')


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def fingerprint(path):
    path = Path(path)
    return dict(path=str(path), bytes=path.stat().st_size, sha256=sha(path))


def verify(records):
    for r in records:
        assert sha(r['path']) == r['sha256'], ('changed input', r['path'])


def retrieval_digest(tasks):
    digest = hashlib.sha256()
    for query, gallery, positive in tasks:
        digest.update(np.asarray([query, len(gallery)], dtype=np.int64).tobytes())
        digest.update(np.asarray(gallery, dtype=np.int64).tobytes())
        digest.update(np.asarray(positive, dtype=np.uint8).tobytes())
    return digest.hexdigest()


def saved_bases():
    bases, paths = {}, []
    for k in KS:
        for kind, filename in [('Intervention-sensitive', 'FAP_residual_PCA'),
                               ('Identity oracle', 'Identity_oracle')]:
            path = V3 / 'bases' / ('HF_stage1_%s_k%d.npy' % (filename, k))
            bases[(kind, k, '')] = np.load(str(path), allow_pickle=False)
            paths.append(path)
        for rep in range(10):
            path = IDAPR / 'A/HF/bases' / ('random_k%d_rep%d.npy' % (k, rep))
            bases[('Random', k, str(rep))] = np.load(str(path), allow_pickle=False)
            paths.append(path)
    for (kind, k, rep), E in bases.items():
        assert E.shape == (256, k), (kind, k, rep, E.shape)
        assert np.isfinite(E).all() and np.allclose(E.T @ E, np.eye(k), atol=1e-9)
    return bases, paths


def make_row(kind, basis, k, rep, domain, stage, n_rois, numerator, denominator, restored):
    assert denominator > 0 and -1e-10 <= numerator <= denominator * (1 + 1e-10)
    energy_pct = 100 * numerator / denominator
    recovery = float(restored['recovered_pct'])
    assert energy_pct > 0, 'Per-energy ratio undefined at zero drift energy'
    return dict(row_type=kind, model='HF', basis=basis, k=k, replicate=rep,
                domain=domain, query_stage=4, gallery_stage=stage, n_rois=n_rois,
                projected_energy_sum=numerator, total_drift_energy_sum=denominator,
                drift_energy_pct=energy_pct, recovered_forgetting_pct=recovery,
                recovery_per_energy=recovery / energy_pct, random_energy_sd='', random_energy_min='',
                random_energy_max='', random_recovery_sd='',
                theory_random_energy_pct=100 * k / 256,
                restoration_source='ics_figs_v3/drift_restoration_HF_native.csv',
                restoration_source_row_type=restored['row_type'])


def main():
    assert mkl.get_max_threads() == 4
    values = dict(line.split('=', 1) for line in (IDAPR / 'inputs.txt').read_text().splitlines() if '=' in line)
    feature_root = Path(next(value.strip() for key, value in values.items() if key.strip() == 'HF_feat_root'))
    bases, basis_paths = saved_bases()
    feature_paths = [feature_root / ('HF_stage%d_e5' % stage) / (domain + '.npz')
                     for stage in [1, 2, 3, 4] for domain in drift.DOMAINS]
    source_paths = [V3 / name for name in ['input_manifest.json', 'split.json', 'retrieval_protocol.json',
                                          'drift_restoration_HF_native.csv']]
    source_paths += [IDAPR / 'inputs.txt', Path(p.__file__), Path(drift.__file__)]
    sources = [fingerprint(path) for path in feature_paths + basis_paths + source_paths]
    original_manifest = {r['path']: r for r in read_json(V3 / 'input_manifest.json')['sources']}
    for record in sources[:len(feature_paths)]:
        assert original_manifest[record['path']]['sha256'] == record['sha256'], ('feature differs from v3', record['path'])
    write_json(ROOT / 'input_manifest.json', dict(sources=sources, bases_reestimated=False,
               basis_orthogonalization_applied=False, feature_hashes_identical_to_v3=True))
    stages = {}
    for stage in [1, 2, 3, 4]:
        stages[stage], _ = drift.load(feature_root, 'HF_stage%d_e5' % stage, False)
        assert np.all(stages[stage]['category'] == 1)
        assert np.isfinite(stages[stage]['Z']).all()
        assert np.allclose(np.linalg.norm(stages[stage]['Z'], axis=1), 1, atol=1e-9)
    ref = stages[1]
    for stage in [2, 3, 4]:
        for key in ['video', 'frame_id', 'instance_id', 'image_id', 'bbox', 'category', 'weather']:
            assert np.array_equal(ref[key], stages[stage][key]), (stage, key)
    fit = p.split_mask(ref['video'], 0)
    split = read_json(V3 / 'split.json')
    assert sorted(np.unique(ref['video'][~fit]).tolist()) == sorted(split['test_videos'])
    assert sorted(np.unique(ref['video'][fit]).tolist()) == sorted(split['fit_videos'])
    assert int(fit.sum()) == split['fit_rois'] and int((~fit).sum()) == split['test_rois']
    restoration = read_csv(V3 / 'drift_restoration_HF_native.csv')
    source_index = {(r['row_type'], r['basis'], int(r['k']), r['replicate'], r['domain']): r for r in restoration}
    old_protocol = read_json(V3 / 'retrieval_protocol.json')
    domain_order = drift.TARGETS[:3]
    contexts, cohorts = {}, []
    for stage, domain in enumerate(domain_order, 1):
        mask = ~fit & (ref['category'] == 1) & (ref['weather'] == domain)
        indices = np.flatnonzero(mask)
        D = stages[4]['Z'][mask] - stages[stage]['Z'][mask]
        total = float(np.sum(D ** 2))
        assert total > 0
        tasks = p.build_retrieval(ref['video'][mask], ref['frame_id'][mask], ref['instance_id'][mask], 4, 30, 4000, 0)
        old = next(row for row in old_protocol['domains'] if row['domain'] == domain)
        assert len(indices) == int(old['test_rois']) and len(tasks) == int(old['queries'])
        task_sha = retrieval_digest(tasks)
        assert task_sha == old['task_sha256'], domain
        cohorts.append(dict(domain=domain, gallery_stage=stage, n_rois=len(D),
                            n_videos=len(np.unique(ref['video'][mask])), total_drift_energy_sum=total,
                            global_roi_indices_sha256=hashlib.sha256(indices.astype(np.int64).tobytes()).hexdigest(),
                            same_v3_retrieval_task_sha256=task_sha, retrieval_queries=len(tasks),
                            all_test_RoIs_used=True))
        contexts[domain] = (stage, D, total)
        log('PAIRS %s stage%d->4 RoIs=%d total_energy=%.12f' % (domain, stage, len(D), total))
    rows = []
    for (basis, k, rep), E in bases.items():
        domain_rows = []
        for domain in domain_order:
            stage, D, total = contexts[domain]
            projected = D @ E
            numerator = float(np.sum(projected ** 2))
            kind = 'random_replicate_domain' if basis == 'Random' else 'domain'
            restored = source_index[(kind, basis, k, rep, domain)]
            row = make_row(kind, basis, k, rep, domain, stage, len(D), numerator, total, restored)
            domain_rows.append(row)
            rows.append(row)
        kind = 'random_replicate_summary' if basis == 'Random' else 'summary'
        restored = source_index[(kind, basis, k, rep, 'three_old_domains')]
        rows.append(make_row(kind, basis, k, rep, 'three_old_domains', '',
                    sum(r['n_rois'] for r in domain_rows), sum(r['projected_energy_sum'] for r in domain_rows),
                    sum(r['total_drift_energy_sum'] for r in domain_rows), restored))
    checks = []
    # A predeclared broad sanity check, not a significance test: theoretical
    # isotropic expectation must lie within the ten-basis empirical mean +/-2SD.
    for k in KS:
        for domain in domain_order + ['three_old_domains']:
            scope = 'summary' if domain == 'three_old_domains' else 'domain'
            rr = [r for r in rows if r['row_type'] == 'random_replicate_' + scope and r['k'] == k and r['domain'] == domain]
            assert len(rr) == 10 and sorted(int(r['replicate']) for r in rr) == list(range(10))
            energy = np.array([r['drift_energy_pct'] for r in rr])
            recovery = np.array([r['recovered_forgetting_pct'] for r in rr])
            efficiency = np.array([r['recovery_per_energy'] for r in rr])
            restored = source_index[(scope, 'Random', k, '', domain)]
            row = make_row(scope, 'Random', k, '', domain, rr[0]['gallery_stage'], rr[0]['n_rois'],
                           float(np.mean([r['projected_energy_sum'] for r in rr])),
                           rr[0]['total_drift_energy_sum'], restored)
            assert abs(row['drift_energy_pct'] - energy.mean()) < 1e-10
            assert abs(row['recovered_forgetting_pct'] - recovery.mean()) < 1e-10
            row.update(drift_energy_pct=float(energy.mean()), random_energy_sd=float(energy.std(ddof=0)),
                       random_energy_min=float(energy.min()), random_energy_max=float(energy.max()),
                       random_recovery_sd=float(recovery.std(ddof=0)),
                       random_replicate_efficiency_mean=float(efficiency.mean()),
                       random_replicate_efficiency_sd=float(efficiency.std(ddof=0)))
            assert abs(row['random_recovery_sd'] - float(restored['random_sd'])) < 1e-9
            rows.append(row)
            theory = 100 * k / 256
            checks.append(dict(domain=domain, k=k, random_mean_pct=float(energy.mean()),
                               random_sd_pct=float(energy.std(ddof=0)), theoretical_pct=theory,
                               difference_pp=float(energy.mean()-theory), relative_error_pct=float(100*(energy.mean()/theory-1)),
                               theoretical_within_mean_plus_minus_2SD=bool(abs(energy.mean()-theory) <= 2 * energy.std(ddof=0)),
                               interpretation='Descriptive numerical sanity check, not a significance test'))
    summary = [next(r for r in rows if r['row_type'] == 'summary' and r['basis'] == basis and r['k'] == k)
               for basis in BASIS_ORDER for k in KS]
    assert len(rows) == 156 and len(summary) == 9
    assert sum(cohort['n_rois'] for cohort in cohorts) == 8117
    for basis in ['Intervention-sensitive', 'Identity oracle']:
        ordered = [r['drift_energy_pct'] for r in summary if r['basis'] == basis]
        assert np.all(np.diff(ordered) >= -1e-10)
    write_csv(ROOT / 'drift_energy_HF_native.csv', rows)
    write_csv(ROOT / 'energy_summary.csv', summary)
    write_csv(ROOT / 'paired_cohorts.csv', cohorts)
    write_csv(ROOT / 'random_theory_selfcheck.csv', checks)
    write_json(ROOT / 'protocol.json', dict(model='HF', query_stage=4, domains=domain_order,
        gallery_stages=[1, 2, 3], category='car', split='Original v3 test videos, seed=0',
        energy_rois=8117, energy_queries_capped=False,
        delta='L2-normalized HF stage4 clean feature minus same-RoI HF stage_j clean feature',
        energy='100 * sum_over_domains_and_RoIs(||delta @ E||^2) / sum_over_domains_and_RoIs(||delta||^2)',
        summation='Ratio of sums; no averaging of per-RoI or per-domain ratios',
        native_bases='Saved HF stage1 PCA/identity bases from ics_figs_v3/bases; unchanged',
        random_bases='Saved idapr/A/HF/bases/random_k{k}_rep{0..9}.npy; unchanged',
        bases_reestimated=False, basis_orthogonalization_applied=False, random_replicates=10, sd_ddof=0,
        restoration='Read directly from ics_figs_v3/drift_restoration_HF_native.csv, including each random replicate',
        efficiency='recovered_forgetting_pct / drift_energy_pct; main-table Random is ratio of means',
        additional_random_efficiency='Mean/SD of paired replicate ratios also retained in CSV, distinguished from the main-table ratio of means',
        random_sanity_rule='Theoretical k/256 within empirical random mean +/-2SD; descriptive, not a significance test',
        no_GPU=True, no_training=True, input_base_commit=BASE_COMMIT))
    write_report(summary, cohorts, checks)
    verify(sources)
    write_json(ROOT / 'complete.json', dict(status='PASS', feature_files=len(feature_paths),
               saved_basis_files=len(basis_paths), paired_test_RoIs=8117, input_hashes_unchanged=True,
               original_v3_restoration_values_unchanged=True, energy_rows=len(rows), summary_rows=len(summary),
               all_combined_random_checks=all(r['theoretical_within_mean_plus_minus_2SD'] for r in checks
                                               if r['domain'] == 'three_old_domains'),
               CPU_only=True, bases_reestimated=False,
               python=platform.python_version(), numpy=np.__version__, MKL_runtime_threads=mkl.get_max_threads()))
    log('ENERGY_ANALYSIS_COMPLETE')


def write_report(summary, cohorts, checks):
    index = {(r['basis'], r['k']): r for r in summary}
    lines = ['# HF 原生漂移能量与遗忘恢复', '',
        '只读 HF 已有特征与已保存基底，纯 CPU；全部新增文件在本目录。', '',
        '每格顺序为 **漂移能量 E% / 恢复遗忘 R%；η=R/E**。'
        'Random 的 E、R 各报告十个固定基底的均值 ± 总体 SD（ddof=0）；'
        '主表 η 是 R 均值除以 E 均值。', '',
        '| 基底 | k=2 | k=8 | k=16 |', '|---|---:|---:|---:|']
    for basis in BASIS_ORDER:
        cells = [basis]
        for k in KS:
            r = index[(basis, k)]
            if basis == 'Random':
                cells.append('%.3f ± %.3f / %.3f ± %.3f；η=%.3f' %
                             (r['drift_energy_pct'], r['random_energy_sd'], r['recovered_forgetting_pct'],
                              r['random_recovery_sd'], r['recovery_per_energy']))
            else:
                cells.append('%.3f / %.3f；η=%.3f' %
                             (r['drift_energy_pct'], r['recovered_forgetting_pct'], r['recovery_per_energy']))
        lines.append('| ' + ' | '.join(cells) + ' |')
    lines += ['', '**η 是两种归一化比例的商，不是因果效应或绝对 mAP 的物理导数。**'
              '恢复遗忘仍是跨阶段身份检索 mAP 损失的代理。Random 每个配对 replicate 的 η、'
              '以及十个 η 的均值/SD 另存 CSV；它们与主表的均值之商不混用。', '',
              '## 计算口径与配对', '',
              '沿用 [v3 split](../ics_figs_v3/split.json) 的测试视频和 car 类别，'
              '每个 stage 的 video/frame/instance/image/bbox/category/weather 均逐行严格相等。'
              '仅使用三个旧域的 **8,117 个一一对应 RoI**；不把检索 query cap 当作能量采样。'
              '检索任务哈希也与 v3 完全一致。', '',
              '| 旧域 | 旧阶段 → query 阶段 | 测试 RoI 对 | 测试视频 | 漂移平方和 | 原检索 queries |',
              '|---|---|---:|---:|---:|---:|']
    for c in cohorts:
        lines.append('| %s | %d → 4 | %d | %d | %.12f | %d |' %
                     (c['domain'], c['gallery_stage'], c['n_rois'], c['n_videos'],
                      c['total_drift_energy_sum'], c['retrieval_queries']))
    lines += ['', '`d = z_stage4 - z_stage_j`，两侧均用与 v3 相同的 L2 归一化 clean 特征。', '',
              '`E% = 100 × Σ三个旧域ΣRoI ||d @ E_basis||² / Σ三个旧域ΣRoI ||d||²`。'
              '先合并分子、分母再相除；不平均逐域或逐 RoI 的比例。', '',
              'HF 原生 Intervention-sensitive 与 Identity oracle 完全复用 '
              '[v3/bases](../ics_figs_v3/bases/) 的六个 stage1 基底；'
              '随机使用 [A2/HF 已保存基底](../idapr/A/HF/bases/) 的三十个 `.npy`。'
              '没有重新估计、重新抽样、QR 或改写基底。全部二十个 HF 特征 NPZ 的哈希与 v3 原记录一致。', '',
              '恢复比例直接取自 [v3 原始 CSV](../ics_figs_v3/drift_restoration_HF_native.csv)，'
              '不重新推理或重算检索指标。', '',
              '## 随机理论值自检', '',
              '均匀随机 k 维正交子空间的期望能量比例为 k/256：'
              '**0.78125% / 3.125% / 6.25%**。十次基底的实测均值不会严格等于期望。'
              '预先采用宽松的数值诊断：理论值在十基底均值 ±2SD 内；这不是显著性检验。', '',
              '| k | 合并随机能量均值 ± SD (%) | k/256 (%) | 实测−理论 (pp) | 相对差 (%) | ±2SD 诊断 |',
              '|---:|---:|---:|---:|---:|---|']
    for c in checks:
        if c['domain'] == 'three_old_domains':
            lines.append('| %d | %.6f ± %.6f | %.6f | %+.6f | %+.3f | %s |' %
                         (c['k'], c['random_mean_pct'], c['random_sd_pct'], c['theoretical_pct'],
                          c['difference_pp'], c['relative_error_pct'],
                          'PASS' if c['theoretical_within_mean_plus_minus_2SD'] else 'CHECK'))
    lines += ['', '所有逐域随机诊断也保留在 [random_theory_selfcheck.csv](random_theory_selfcheck.csv)。'
              '该诊断不会导致重新选基底或更改数值。', '',
              '## 文件与复现', '',
              '[drift_energy_HF_native.csv](drift_energy_HF_native.csv) 包含 156 行：'
              '27 个逐域均值/确定性结果、9 个合并结果、90 个逐域随机 replicate 和 30 个合并随机 replicate。'
              '投影平方和、总漂移平方和、恢复比例、效率、随机 SD/min/max 全部保留。', '',
              '输入与源码哈希见 [input_manifest.json](input_manifest.json)，'
              '配对计数和索引/任务哈希见 [paired_cohorts.csv](paired_cohorts.csv)，'
              '汇总表原始数据见 [energy_summary.csv](energy_summary.csv)。'
              '详细定义见 [protocol.json](protocol.json)。', '',
              '使用 [README.txt](README.txt) 的 Python 与四线程设置运行 '
              '[scripts/run_all.sh](scripts/run_all.sh)。无 GPU、无训练，历史实验目录不改动。', '']
    (ROOT / 'SUMMARY.md').write_text('\n'.join(lines).rstrip() + '\n')


if __name__ == '__main__':
    main()
