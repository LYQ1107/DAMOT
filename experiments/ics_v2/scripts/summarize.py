"""APR-off C-series table and the final, predeclared single-seed criterion."""
from common import *
import statistics
import subprocess

GROUPS = ['X3-1','X3-0','C0','C1','C2','C3','C4']
METRICS = ['HOTA','DetA','IDF1','AssA']
rows = []
for group in GROUPS:
    for domain in ['night','dawn_dusk']:
        cell = Q / 'evaluation' / group / domain
        if not (cell / 'complete.json').exists():
            continue
        for r in csv.DictReader((cell / 'metrics.csv').open()):
            rows.append(dict(r, group=group, APR_enabled=group.startswith('X3'), training_seed=1790355234))
if rows:
    write_csv(Q / 'metrics.csv', rows)
overall = {(r['group'],r['domain']):r for r in rows if r['category']=='OVERALL'}
audits, activation = [], []
for group in ['C0','C1','C2','C3','C4']:
    root = Q / 'runs' / group
    for rank in range(4):
        p = root / ('v2_end_rank%d.json' % rank)
        if p.exists():
            audits.append(dict(json.loads(p.read_text()), group=group))
        p = root / ('v2_iteration50_rank%d.json' % rank)
        if p.exists():
            activation.append(dict(json.loads(p.read_text()), group=group))
if audits:
    write_csv(Q / 'training_audit.csv', audits)
if activation:
    write_csv(Q / 'activation_first50.csv', activation)
comparisons = []
if all((g,'night') in overall for g in ['C0','C1','C2','C3','C4']):
    for group in ['C1','C2','C3']:
        for metric in ['HOTA','AssA']:
            delta = float(overall[group,'night'][metric+'_percent']) - float(overall['C0','night'][metric+'_percent'])
            random_delta = float(overall[group,'night'][metric+'_percent']) - float(overall['C4','night'][metric+'_percent'])
            passed = delta >= .4 and (group == 'C1' or random_delta > 0)
            comparisons.append(dict(group=group,metric=metric,delta_vs_C0=delta,
                  delta_vs_C4=random_delta,threshold_vs_C0=.4,requires_same_metric_above_C4=group!='C1',
                  metric_criterion_passed=passed))
    write_csv(Q / 'criterion.csv', comparisons)
complete = len(rows)==98 and len(audits)==20 and len(comparisons)==6
lines = ['# ICS v2: APR-off single-seed Night experiments', '',
 '**Training seed: 1790355234 only.** C0–C4 all disable APR and do not load APR history. '
 'Each starts from the identical HF stage3 learned weights, trains Night for two epochs, four V100s, '
 '16 samples/GPU, two workers/GPU. APR-off refers to this Night continuation; the common HF stage3 '
 'prefix is retained. All five set causal_min_samples=32 and roi_proto_couple_ics=False.', '',
 'Status: '+('complete.' if complete else 'partial; pending requested cells.'), '',
 'C0: no ICS. C1: old global_pc_ema, one PC, alpha=.25, loss weight=.1, bank reset to constructor state '
 '(student and teacher); all nonbank state is retained. C2/C3: frozen A-stage HF stage3 fit-video GT-box '
 'FAP nuisance-2, alpha=.25/.5, weight=.1. C4: frozen seed0 random orthogonal nuisance control, alpha=.25, weight=.1. '
 'C2–C4 store rows [2,256], initialized=True, and make zero bank update calls.', '',
 'X3-1/X3-0 are reused APR-on references, not controls for the revised criterion. '
 'X3-1 disables ICS; X3-0 uses the final original ICS settings (alpha=.1, weight=.025, min_samples96). '
 'The C-series inference retains the X3 frozen student, Conv-BN fusion, thresholds .05/.30/.20, '
 'full 36-video/7286-frame Night and 33-video/6476-frame Dawn/Dusk pools, all-frame HOTA.', '',
 '| Group | APR | Domain | HOTA | DetA | IDF1 | AssA | Night IDSW |',
 '|---|---|---|---:|---:|---:|---:|---:|']
for group in GROUPS:
    for domain in ['night','dawn_dusk']:
        row = overall.get((group,domain))
        if row is None:
            lines.append('| %s | %s | %s | pending | — | — | — | — |' % (group,'on' if group.startswith('X3') else 'off',domain))
            continue
        values = ['%.6f'%float(row[m+'_percent']) for m in METRICS]
        values.append(str(int(float(row['IDSW']))) if domain=='night' else '—')
        lines.append('| %s | %s | %s | %s |' % (group,'on' if group.startswith('X3') else 'off',domain,' | '.join(values)))
lines += ['', 'Predeclared revised criterion: C1 is effective if Night HOTA **or** AssA exceeds C0 by at '
          'least 0.4 points. C2/C3 must also exceed C4 in the same qualifying metric. '
          'This criterion replaces the initial APR-on/X3 comparison rule. It is descriptive for one seed, '
          'not a statistical significance claim.', '']
for group in ['C1','C2','C3']:
    group_rows = [r for r in comparisons if r['group']==group]
    if group_rows:
        lines.append('- %s: %s. %s.' % (group,'criterion met' if any(r['metric_criterion_passed'] for r in group_rows) else 'criterion not met',
          '; '.join('%s ΔC0=%+.6f, ΔC4=%+.6f'%(r['metric'],r['delta_vs_C0'],r['delta_vs_C4']) for r in group_rows)))
lines += ['', 'Training activity and startup verification:', '']
for group in ['C0','C1','C2','C3','C4']:
    records = [r for r in audits if r['group']==group]
    if len(records)==4:
        n = sum(r['total_batches'] for r in records)
        key = sum(r['key_effective_batches'] for r in records)/n
        ref = sum(r['ref_effective_batches'] for r in records)/n
        losses = [r['loss_track_trim_mean'] for r in records if r['loss_track_trim_mean'] is not None]
        count = sum(r['trim_active_batches'] for r in records)
        mean = ('absent (ICS off)' if not count else '%.9f'%(sum(
            (r['loss_track_trim_mean'] or 0.)*r['trim_active_batches'] for r in records)/count))
        first = ('fixed at startup; zero updates' if records[0]['fixed_basis'] else
                 ([r['first_initialized_iteration'] for r in records] if group!='C0' else 'not applicable'))
        lines.append('- %s: 228 batches/rank, key/ref actual nonidentity ratios %.4f/%.4f; '
                     'mean weighted loss_track_trim %s; first initialization by rank %s.' % (group,key,ref,mean,first))
lines += ['', 'The actual nonidentity ratios count all training batches, independently for key and ref. '
          'A trim call returning the original values is inactive even when a trim loss is logged. '
          'The exact first50 records and any below80% alerts are in activation_first50.csv and per-rank logs.', '',
          'CPU preflight verifies learned weights and all nonbank state bitwise, C1 bank-only replacement, '
          'the 32-RoI gate, finite gradients, exact operator orientation, row orthonormal error <1e-5 '
          'and principal angles <0.01°. Formal per-rank audits verify the same startup state, zero APR '
          'history and zero fixed-bank update calls. Check basis/provenance.json, basis/A1_selected_entry.json, '
          'runs/*/config_diff.json, manifests/preflight.json and manifests/final_validation.json.', '',
          'The five evaluated server checkpoints, their SHA256 values and metric hashes are recorded '
          'in manifests/final_model_provenance.json; checkpoint weights stay on the server.', '',
          'Historical full-run reassessment (separate legacy evaluator): '
          '[completed comparison](historical_completed/SUMMARY.md), '
          '[all stage/domain metrics](historical_completed/stage_domain_metrics.csv), '
          '[old-ICS minus no-ICS deltas](historical_completed/old_ICS_comparisons.csv). '
          '[Per-domain diagonal HOTA/AssA](historical_completed/DIAGONAL_BY_DOMAIN.md) and '
          '[C1 exact activity counts](manifests/C1_activity_summary.json) are included. '
          'Twelve missing old-domain cells were inferred; no historical training was repeated. '
          'Historical seeds, APR strengths and batch/GPU counts differ and are disclosed. '
          'The initial pre-inference table is preserved in historical/SUMMARY.md.', '',
          'Only this task directory was modified. The starting snapshots are read-only symlinks to the '
          'identical original HF file. Redundant task-owned epoch1/stage00 copies and duplicate '
          'COCO exports are removed with hash records; epoch1 recovery files remain until training '
          'succeeds, while unused stage00 copies may be removed earlier for disk space. '
          'Epoch2 and the final endpoint remain on the server. '
          'Weights, NPZs and raw predictions are excluded from GitHub.', '']
(Q / 'SUMMARY.md').write_text('\n'.join(lines))
if complete:
    subprocess.run([sys.executable,str(Q / 'scripts/report_diagonals.py')],check=True)
dump(Q / 'manifests/result_status.json', dict(complete=complete,metric_rows=len(rows),training_ranks=len(audits),
     evaluation_cells=len(overall),required_evaluation_cells=14,groups=['C0','C1','C2','C3','C4'],APR_enabled=False,seed=1790355234))
print('SUMMARY',len(rows),len(audits),complete,flush=True)
