"""Read archived logs/configs only. Never infer or fill missing cells."""
from common import *
import statistics
import shutil
import re
from mmcv import Config

B = Path('/data1/liuyeqiang/idapr_20261030_153059/B3stage/nas/A100/data1/LWR/vranlee/SERVER_ONLY/avis/DA-MOT/darth-main/work_dirs/bdd_4domain')
DOMS = ['overcast', 'rainy', 'dawn_dusk', 'night']
METRICS = ['HOTA', 'DetA', 'IDF1', 'AssA']
RUNS = {
 'SPA_pipeline': 'spa_fix/spa_pipeline_mild_4domain_5e',
 'SPA_legacy': 'spa_only_4domain_5e',
 'SPA_oldICS': 'module_combinations/spaP1_causalOld_4domain_5e',
 'SPA_APR': 'module_combinations/spaP1_roiV1_4domain_5e',
 'F_old': 'module_combinations/spaP1_causalOld_roiV1Weak_4domain_5e',
 'causalShrinkage': 'tri_module_pca_variants/spaP1_causalShrinkage_roiV1Weak_4domain_5e',
 'causalProtoCenter': 'tri_module_pca_variants/spaP1_causalProtoCenter_roiV1Weak_4domain_5e',
 'causalProtoShrink': 'tri_module_pca_variants/spaP1_causalProtoShrink_roiV1Weak_4domain_5e',
}
HEADERS = ['IDF1 MOTA MOTP FP FN IDSw Rcll Prcn MT PT ML FM',
           'AssRe AssPr AssA LocA DetRe DetPr DetA HOTA Dets IDs']
provenance = []


def retain(source, target):
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(str(source), str(target))
    provenance.append(dict(source=str(source), local=str(target.relative_to(Q)), sha256=sha(target)))


def parse_log(path):
    lines = path.read_text().splitlines()
    result = {}
    for header in HEADERS:
        for i, line in enumerate(lines):
            if ' '.join(line.split()) == header:
                for item in lines[i + 1:i + 20]:
                    if item.strip().startswith('OVERALL'):
                        result.update({k: float(v) for k, v in zip(header.split(), item.split()[1:])})
                        break
                break
    summaries = list(path.parent.glob('qdtrack_*/*/summary_metrics.csv'))
    assert len(summaries) <= 1, (path, summaries)
    if summaries:
        full = next(r for r in csv.DictReader(summaries[0].open()) if r[''] == 'OVERALL')
        for key in METRICS:
            result[key] = float(full[key])
        result['_summary_path'] = str(summaries[0])
    return result


rows, configs, summaries, gaps = [], [], [], []
for label, relative in RUNS.items():
    root = B / relative
    method = root.name
    seeds = []
    for stage, domain in enumerate(DOMS, 1):
        stage_name = 'stage%02d_after_%s' % (stage, domain)
        train = root / 'episode/adapt' / stage_name
        paths = list(train.glob('darth_*.py'))
        assert len(paths) == 1, (label, stage, paths)
        local = Q / 'historical/configs' / label / (stage_name + '.py')
        retain(paths[0], local)
        cfg = Config.fromfile(str(local))
        head = cfg.model.track_head
        causal = {k: v for k, v in head.items() if k.startswith(('causal_', 'roi_proto_')) or k in ['use_causal_direction_bank', 'use_roi_proto_memory']}
        assert cfg.total_epochs == 5
        assert cfg.get('runner', {}).get('max_epochs', 5) == 5
        logs = list(train.glob('*.log.json'))
        epochs = []
        logged_seeds = []
        all_log_seeds = []
        training_log_seeds = []
        assert logs
        for log in sorted(logs):
            retain(log, Q / 'historical/train_logs' / label / stage_name / log.name)
            values = [json.loads(line) for line in log.read_text().splitlines()]
            trained = [v for v in values if v.get('mode') == 'train']
            epochs.extend(v['epoch'] for v in trained)
            header_seeds = [v['seed'] for v in values if 'seed' in v]
            all_log_seeds.extend(header_seeds)
            if trained:
                training_log_seeds.extend(header_seeds)
            if any(v['epoch'] == 5 for v in trained):
                logged_seeds.extend(header_seeds)
        epoch_evidence = 'per-stage JSON train log'
        if not epochs or max(epochs) != 5:
            proved = False
            for episode_log in list((root / 'logs').glob('train_episode.log')) + list((root / 'episode').glob('episode_*.log')):
                text = episode_log.read_text()
                starts = list(re.finditer(r'\[STAGE (\d+)\] (?:Start from|Continue from).*target=([^\s]+)', text))
                for pos, match in enumerate(starts):
                    if int(match.group(1)) == stage and match.group(2) == domain:
                        segment = text[match.start():starts[pos + 1].start() if pos + 1 < len(starts) else len(text)]
                        if 'Saving checkpoint at 5 epochs' in segment:
                            retain(episode_log, Q / 'historical/episode_logs' / label / episode_log.name)
                            epochs.append(5)
                            epoch_evidence = 'stage-scoped episode log: checkpoint saved at epoch5'
                            proved = True
                            break
                if proved:
                    break
        assert epochs and max(epochs) == 5 and (train / 'epoch_5.pth').exists(), (label, stage, epochs)
        logged_seeds = logged_seeds or training_log_seeds or all_log_seeds
        seeds.append(cfg.seed)
        configs.append(dict(run=label, stage=stage, method=method, source=str(paths[0]), seed=cfg.seed,
                            track_head_type=head.type,
                            epochs=5, logged_max_epoch=max(epochs), samples_per_gpu=cfg.data.samples_per_gpu,
                            actual_training_seeds=sorted(set(logged_seeds)),
                            all_startup_metadata_seeds=sorted(set(all_log_seeds)),
                            auxiliary_startup_seeds=sorted(set(all_log_seeds)-set(logged_seeds)),
                            epoch_evidence=epoch_evidence,
                            resolved_seed_matches_logs=set(logged_seeds) == {cfg.seed},
                            gpu_ids=list(cfg.gpu_ids), causal_roi_proto=causal,
                            student_pipeline=cfg.model.get('student_pipeline'),
                            use_spa_frequency_aug=cfg.model.get('use_spa_frequency_aug')))
    assert len(set(seeds)) == 1
    look = {}
    for stage, learned in enumerate(DOMS, 1):
        stage_name = 'stage%02d_after_%s' % (stage, learned)
        for domain in DOMS:
            options = [root / 'forgetting_eval' / stage_name / 'epoch_5' / domain / 'eval.log',
                       B / 'continual_forgetting_eval' / method / stage_name / 'epoch_5' / domain / 'eval.log',
                       root / 'eval' / stage_name / 'epoch_5' / domain / 'eval.log']
            candidates = [(p, parse_log(p)) for p in options if p.exists()]
            candidates = [(p, m) for p, m in candidates if all(k in m for k in METRICS)]
            if not candidates:
                gaps.append(dict(run=label, stage=stage, domain=domain, status='not available; not rerun'))
                continue
            source, metrics = candidates[0]
            retain(source, Q / 'historical/eval_logs' / label / stage_name / domain / 'eval.log')
            if metrics.get('_summary_path'):
                retain(Path(metrics['_summary_path']), Q / 'historical/eval_logs' / label / stage_name / domain / 'summary_metrics.csv')
            look[stage, domain] = metrics
            rows.append(dict(run=label, method=method, stage=stage, domain=domain, epoch=5, seed=seeds[0],
                             protocol='archived legacy evaluator', precision='full existing CSV' if metrics.get('_summary_path') else 'printed log', source=str(source),
                             **{m: metrics[m] for m in METRICS}, IDSW=metrics.get('IDSw')))
    for metric in METRICS:
        def avg(keys):
            return statistics.mean(look[k][metric] for k in keys) if all(k in look for k in keys) else ''
        final_mean = avg([(4, d) for d in DOMS])
        final_old = avg([(4, d) for d in DOMS[:3]])
        diagonal = avg([(j, d) for j, d in enumerate(DOMS, 1)])
        seen = [avg([(s, d) for d in DOMS[:s]]) for s in range(1, 5)]
        aip = statistics.mean(seen) if all(v != '' for v in seen) else ''
        needed = [(s, DOMS[j - 1]) for j in range(1, 4) for s in range(j, 5)]
        if all(k in look for k in needed):
            forgetting = statistics.mean(max(look[s, DOMS[j - 1]][metric] for s in range(j, 4)) - look[4, DOMS[j - 1]][metric] for j in range(1, 4))
            bwt = statistics.mean(look[4, DOMS[j - 1]][metric] - look[j, DOMS[j - 1]][metric] for j in range(1, 4))
        else:
            forgetting, bwt = '', ''
        summaries.append(dict(run=label, metric=metric, seed=seeds[0], epochs='4 x 5', observed_cells=len(look),
                              required_cells=16, FinalMean=final_mean, AIP=aip, AvgF=forgetting,
                              FinalOld=final_old, BWT=bwt, DiagonalMean=diagonal))
write_csv(Q / 'historical/stage_domain_metrics.csv', rows)
write_csv(Q / 'historical/continual_metrics.csv', summaries)
write_csv(Q / 'historical/missing_cells.csv', gaps)
dump(Q / 'historical/resolved_config_audit.json', configs)
dump(Q / 'historical/provenance.json', provenance)
comparisons = []
for treatment, control in [('SPA_oldICS', 'SPA_pipeline'), ('F_old', 'SPA_APR')]:
    for metric in METRICS:
        a = next(r for r in summaries if r['run'] == treatment and r['metric'] == metric)
        b = next(r for r in summaries if r['run'] == control and r['metric'] == metric)
        comparisons.append(dict(ICS=treatment, no_ICS=control, metric=metric, same_seed=a['seed'] == b['seed'],
                                ICS_seed=a['seed'], control_seed=b['seed'],
                                delta_FinalMean=a['FinalMean'] - b['FinalMean'] if a['FinalMean'] != '' and b['FinalMean'] != '' else '',
                                delta_AvgF=a['AvgF'] - b['AvgF'] if a['AvgF'] != '' and b['AvgF'] != '' else '',
                                limitation='No-ICS controls lack final old-domain evaluations; no results filled. APR strengths may also differ.'))
write_csv(Q / 'historical/old_ICS_comparisons.csv', comparisons)
lines = ['# Existing four-domain five-epoch runs (no training or inference)', '',
 'Uniform archived legacy evaluator; original full-precision CSVs are used when available (per-cell provenance records precision). Do not mix these values with the all-frame HOTA '
 'of the new X3-protocol runs.', '',
 'AIP = mean of four seen-domain stage means. FinalMean = stage4 four-domain mean. FinalOld = stage4 '
 'three old-domain mean. AvgF = mean(best prior learned-stage value, excluding stage4, minus stage4), '
 'retaining negative values. BWT = mean(stage4 minus diagonal) over three old domains. '
 'DiagonalMean is diagnostic; it is never substituted for FinalMean.', '',
 '| Run | Metric | Seed | Cells | FinalMean | AIP | AvgF | FinalOld | BWT |',
 '|---|---|---:|---:|---:|---:|---:|---:|---:|']
for r in summaries:
    values = ['N/A' if r[k] == '' else '%.6f' % r[k] for k in ['FinalMean','AIP','AvgF','FinalOld','BWT']]
    lines.append('| %s | %s | %s | %s/16 | %s |' % (r['run'],r['metric'],r['seed'],r['observed_cells'],' | '.join(values)))
lines += ['', '**Old ICS minus no ICS: FinalMean and AvgF differences are unavailable.** The no-ICS '
          'runs have four diagonal evaluations and no stage4 old-domain results. No missing run or '
          'evaluation was added. Available and missing cells are listed separately in the CSVs.', '']
for r in comparisons:
    if r['metric'] == 'HOTA':
        lines.append('- %s vs %s: seeds %s vs %s; same seed = %s.' % (r['ICS'],r['no_ICS'],r['ICS_seed'],r['control_seed'],r['same_seed']))
lines += ['', 'SPA_pipeline is the pipeline FAP baseline. SPA_legacy uses a different augmentation '
          'implementation and is separate. SPA_APR uses roiV1; F_old uses roiV1Weak, so APR is also '
          'not a controlled match. Every stage resolved config, causal/ROI fields, batch/GPU count, seed '
          'and epoch5 train log is retained with SHA256 provenance. Training completion and evaluation '
          'coverage are separate.', '']
(Q / 'historical/SUMMARY.md').write_text('\n'.join(lines))
dump(Q / 'manifests/historical_complete.json', dict(status='complete', training_or_inference_started=False,
     runs=len(RUNS), cells=len(rows), missing_cells=len(gaps), all_stages_epoch5_verified=True,
     missing_results_not_filled=True))
print('HISTORICAL_COMPLETE', len(RUNS), len(rows), len(gaps), flush=True)
