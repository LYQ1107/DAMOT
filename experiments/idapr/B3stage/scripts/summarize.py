"""One requested comparison table, all per-class metrics and transparent audits."""
from common import *
import statistics

METRICS = ['HOTA', 'DetA', 'IDF1', 'AssA']
OLD = ['overcast', 'rainy', 'dawn_dusk']
DOMAINS = ['night'] + OLD


def read(path):
    return list(csv.DictReader(path.open()))


def union_csv(path, rows):
    if rows:
        keys = list(dict.fromkeys(k for r in rows for k in r))
        write_csv(path, [{key: r.get(key, '') for key in keys} for r in rows])


baseline = {}
for domain in OLD:
    path = Q / 'evaluation/HF_stage3' / domain / 'metrics.csv'
    if path.exists():
        for row in read(path):
            baseline[(domain, row['category'])] = row
rows = []
for group in ['B0', 'R', 'B3']:
    for domain in DOMAINS:
        cell = Q / 'evaluation' / group / domain
        if not (cell / 'complete.json').exists():
            continue
        for source in read(cell / 'metrics.csv'):
            row = dict(source, group=group, domain=domain, training_seed=1790355234)
            for metric in METRICS:
                row['forgetting_' + metric] = (float(baseline[(domain, row['category'])][metric + '_percent'])
                                               - float(row[metric + '_percent']) if domain in OLD else '')
            rows.append(row)
means = []
for group in ['B0', 'R', 'B3']:
    for category in list(CLASSES) + ['OVERALL']:
        old = [r for r in rows if r['group'] == group and r['category'] == category and r['domain'] in OLD]
        if len(old) == 3:
            means.append(dict(group=group, category=category, old_domains=3,
                              **{'mean_forgetting_' + m: statistics.mean(float(r['forgetting_' + m]) for r in old)
                                 for m in METRICS}))
union_csv(Q / 'metrics.csv', rows)
union_csv(Q / 'mean_forgetting.csv', means)
overall = {(r['group'], r['domain']): r for r in rows if r['category'] == 'OVERALL'}
avg = {r['group']: r for r in means if r['category'] == 'OVERALL'}
comparisons = []
for reference in ['B0', 'R']:
    for domain in DOMAINS:
        if ('B3', domain) in overall and (reference, domain) in overall:
            delta = float(overall[('B3', domain)]['HOTA_percent']) - float(overall[(reference, domain)]['HOTA_percent'])
            comparisons.append(dict(reference=reference, scope=domain, B3_minus_reference_HOTA=delta,
                                    interpretation='no difference' if abs(delta) <= .3 else ('B3 higher' if delta > 0 else 'B3 lower')))
    if 'B3' in avg and reference in avg:
        reduction = float(avg[reference]['mean_forgetting_HOTA']) - float(avg['B3']['mean_forgetting_HOTA'])
        comparisons.append(dict(reference=reference, scope='mean_old_forgetting_reduction',
                                B3_minus_reference_HOTA=reduction,
                                interpretation='no difference' if abs(reduction) <= .3 else ('B3 less forgetting' if reduction > 0 else 'B3 more forgetting')))
union_csv(Q / 'comparison.csv', comparisons)
audit_rows = []
serialization_path = Q / 'U/checkpoint_serialization.json'
serialization = ({r['group']: r for r in json.loads(serialization_path.read_text())['records']}
                 if serialization_path.exists() else {})
for group in ['B3', 'R']:
    for path in (Q / 'runs' / group).glob('apr_audit_end_rank*.json'):
        audit = dict(json.loads(path.read_text()), group=group)
        audit['original_hook_claim_basis_excluded'] = audit.pop('basis_not_persisted_in_checkpoint')
        if group in serialization:
            audit['actual_original_checkpoint_basis_keys'] = len(serialization[group]['original_basis_keys'])
            audit['verified_export_basis_excluded'] = True
        iterations = read(path.parent / ('apr_iterations_rank%d.csv' % audit['rank']))
        values = [float(r['weighted_APR_loss']) for r in iterations if r['weighted_APR_loss'] != '']
        shares = [float(r['mean_in_share']) for r in iterations
                  if r['mean_in_share'] != '' and r['weighted_APR_loss'] != '']
        audit.update(APR_active_batches=len(values), APR_nonzero_batches=sum(v != 0.0 for v in values),
                     weighted_APR_mean=statistics.mean(values) if values else '',
                     weighted_APR_min=min(values) if values else '',
                     mean_in_share=statistics.mean(shares) if shares else '')
        audit_rows.append(audit)
union_csv(Q / 'training_audit.csv', audit_rows)
complete = len(rows) == 84 and len(means) == 21
lines = ['# B3 nuisance release vs native ICS coupling', '',
         '**Single training seed: 1790355234.** B3 and R each train Night for two epochs with four V100s, '
         '16 samples/GPU and 2 workers/GPU. B0 reuses X3-0 without retraining.', '',
         'Status: ' + ('all requested experiments complete.' if complete else 'partial; missing groups remain pending.'), '',
         'This follow-up was separately authorized after the original idAPR identity gate failed. '
         'B1/B2 and multi-seed experiments are outside this task.', '',
         'B3 disables APR–ICS coupling and releases the fixed teacher-estimated nuisance-2 space. '
         'R retains native HF coupling (beta=1, basis=ICS, metric=projected_cosine), with space_mode=full. '
         'Both retain the original ICS settings and the same SHA256-identified three-domain APR history.', '',
         'All values and differences are percentage points. F = HF stage3 old-domain value minus the final value; '
         'AvgF is the equal average of Overcast, Rainy and Dawn/Dusk, retaining negative values. '
         'The four-number tuples use the order **HOTA / DetA / IDF1 / AssA**.', '',
         '| Group | Domain | HOTA | DetA | IDF1 | AssA | F (HOTA / DetA / IDF1 / AssA) | AvgF (same order) |',
         '|---|---|---:|---:|---:|---:|---|---|']
for group in ['B0', 'R', 'B3']:
    for domain in DOMAINS:
        r = overall.get((group, domain))
        if r is None:
            lines.append('| %s | %s | pending | — | — | — | — | — |' % (group, domain))
            continue
        forgetting = ' / '.join('%.6f' % float(r['forgetting_' + m]) for m in METRICS) if domain in OLD else '—'
        average = (' / '.join('%.6f' % float(avg[group]['mean_forgetting_' + m]) for m in METRICS)
                   if group in avg and domain == 'night' else '—')
        lines.append('| %s | %s | %.6f | %.6f | %.6f | %.6f | %s | %s |' %
                     (group, domain, *[float(r[m + '_percent']) for m in METRICS], forgetting, average))
lines += ['', 'Complete per-class values and differences: [metrics.csv](metrics.csv), '
          '[mean_forgetting.csv](mean_forgetting.csv), [comparison.csv](comparison.csv).', '']
if len(comparisons) == 10:
    beyond = [r for r in comparisons if r['interpretation'] != 'no difference']
    if not beyond:
        lines.append('Interpretation: every B3 comparison to B0 and R is within ±0.3 HOTA; '
                     'all are “无差异” under the requested single-seed rule.')
    else:
        lines.append('Comparisons exceeding the 0.3 HOTA threshold (B3 minus reference; '
                     'positive old-forgetting reduction means less forgetting): ' + '; '.join(
                     '%s vs %s: %+.6f (%s)' % (r['scope'], r['reference'], r['B3_minus_reference_HOTA'], r['interpretation'])
                     for r in beyond) + '. All remaining comparisons are “无差异”.')
    lines.append('These are single-seed observations, not estimates of statistical significance.')
else:
    lines.append('Interpretation: pending both formal groups and all requested evaluations; ±0.3 HOTA is “无差异”.')
lines += ['', 'Verification and basis provenance:', '']
for name in ['U1', 'U2', 'extractor_parity', 'warmstart', 'checkpoint_serialization']:
    p = Q / 'U' / (name + '.json')
    lines.append('- %s: %s.' % (name, json.loads(p.read_text())['status'] if p.exists() else 'pending'))
smoke = Q / 'U/U3.json'
if smoke.exists():
    s = json.loads(smoke.read_text())
    lines.append('- U3: %s, 50 updates, APR finite/nonzero in every iteration, history_domains=3. '
                 'Mean weighted APR %.9f; same-step full-space reference %.9f; release/reference ratio %.6f; '
                 'mean in-space share %.6f.' % (s['status'], s['APR_loss_mean'], s['paired_full_reference_mean'],
                                                s['release_to_paired_full_ratio'], s['mean_in_share']))
    lines.append('- Existing B0 log reference: first %d available iterations, mean APR %.9f; '
                 'B3/native-B0 ratio %.6f. This reference is unpaired and uses four GPUs. '
                 'The 50-step same-feature full-space reference above provides the paired scaling check.' %
                 (s['native_B0_reference_iterations'], s['native_B0_APR_mean'], s['release_to_native_B0_ratio']))
else:
    lines.append('- U3: pending.')
info_path = Q / 'basis/E_n_k2.json'
if info_path.exists():
    b = json.loads(info_path.read_text())
    lines.append('- Teacher basis: %d RoIs, %d images, %d videos with proposals, score ≥0.75; '
                 'all instance IDs are -1. The plan covers all 33 Dawn/Dusk videos, 990 sampled frames '
                 'with the original extractor frame sampling.' % (b['rois'], b['images'], b['videos_with_teacher_rois']))
    lines.append('- Basis SHA256: `%s`. Principal angles to A/HF-stage3 nuisance-2: %s degrees. '
                 'A used FAP-only GT-box features on fit videos across domains; this training basis uses '
                 'teacher boxes and three perturbation views on Dawn/Dusk, as newly requested.' %
                 (b['sha256'], ', '.join('%.6f' % a for a in b['principal_angles_to_A_HF_stage3_nuisance2_deg'])))
lines += ['', 'Protocol notes and deviations:', '',
          '- U3 does not retrain B0. Its archived log has 20-step averages, so the native comparison '
          'uses the available first40 steps; an exact 50-step same-feature full-space counterfactual '
          'is additionally recorded. Smoke weights are discarded and are never formal starting weights.',
          '- Default full losses were checked bitwise on the same serialized train batch; GT extraction '
          'was checked on two fixed frames, and teacher extraction was replayed with all GT boxes removed.',
          '- The actual X3 HF runtime is copied under this task. The original HF finite-loss, EMA '
          'and ICS audit hooks are retained; the extra audit only records losses and fixed-basis state.',
          '- Independent read-only SSHFS under this task replaces a stale former private mount. '
          'The current shared mount and every previous task directory remain unchanged.',
          '- Native frozen inference, fusion, thresholds .05/.30/.20 and all-frame HOTA are unchanged. '
          'HF stage3 Overcast/Rainy metrics exactly match all 56 archived class/metric reference values; '
          'Dawn/Dusk baseline and B0 Night/Dawn evaluations are reused as requested.',
          '- After each group finishes all four evaluations, only its own redundant epoch1 and stage00 '
          'source copies are removed. Epoch2, the final endpoint and stage3 starting copy are retained.',
          '- Final file validation detected that this old MMCV deliberately serializes nonpersistent buffers. '
          'The original B3 checkpoint contains two fixed basis buffers, despite the original audit flag '
          '(which checked PyTorch state_dict only). The final code adds a release-only state-dict hook '
          'and checks MMCV directly. The actual training versions of the two affected source files are '
          'preserved in source_training/, with their original pre-training seal and code_training.patch. '
          'No forward, loss or update code changed. The clean B3 epoch_2_export.pth omits only these two '
          'buffers; all 752 retained tensors and all other checkpoint payload are bitwise identical. '
          'Evaluations reference the preserved original checkpoints and remain valid; no training or '
          'inference was repeated for this serialization fix.',
          '', 'See `inputs.txt`, `runs/*/config_diff.json`, `U/`, `basis/`, `training_audit.csv` '
          'and `manifests/` for exact source paths and checksums. Model weights, raw predictions '
          'and feature NPZ files remain on the server.', '']
(Q / 'SUMMARY.md').write_text('\n'.join(lines))
dump(Q / 'manifests/result_status.json', dict(complete=complete, metric_rows=len(rows),
     formal_groups=['B3', 'R'], seed=1790355234, B0_reused=True,
     completed_evaluation_cells=sum((Q / 'evaluation' / g / d / 'complete.json').exists()
         for g in ['B0', 'R', 'B3', 'HF_stage3'] for d in (OLD if g == 'HF_stage3' else DOMAINS)),
     required_evaluation_cells=15))
print('SUMMARY', len(rows), len(means), complete, flush=True)
