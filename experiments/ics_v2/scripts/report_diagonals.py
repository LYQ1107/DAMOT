"""Per-domain diagonal HOTA/AssA and completed C1 activity, CPU CSV only."""
from common import *
import statistics

out = Q / 'historical_completed'
rows = list(csv.DictReader((out / 'stage_domain_metrics.csv').open()))
means = list(csv.DictReader((out / 'continual_metrics.csv').open()))
domains = ['overcast','rainy','dawn_dusk','night']
labels = {
 'SPA_pipeline':'SPA/FAP，无 ICS、无 APR', 'SPA_legacy':'旧版 SPA，无 ICS、无 APR',
 'SPA_oldICS':'SPA/FAP + 老 ICS', 'SPA_APR':'SPA/FAP + APR，无 ICS',
 'F_old':'F_old（老 ICS + APR）', 'causalShrinkage':'causalShrinkage + APR',
 'causalProtoCenter':'causalProtoCenter + APR', 'causalProtoShrink':'causalProtoShrink + APR'}
diagonal = []
lines = ['# Per-domain diagonal scores', '',
 'Each cell is HOTA / AssA, from the original legacy evaluator at the end of its own adaptation stage. '
 'R11=Overcast, R22=Rainy, R33=Dawn/Dusk, R44=Night. Historical seeds differ.', '',
 '| Run | Seed | R11 Overcast | R22 Rainy | R33 Dawn/Dusk | R44 Night |',
 '|---|---:|---:|---:|---:|---:|']
for group,label in labels.items():
    cells = []
    for stage,domain in enumerate(domains,1):
        r = next(r for r in rows if r['run']==group and int(r['stage'])==stage and r['domain']==domain)
        cell = dict(run=group,label=label,seed=r['seed'],stage=stage,domain=domain,
                    HOTA=float(r['HOTA']),AssA=float(r['AssA']),source=r['source'])
        cells.append(cell)
        diagonal.append(cell)
    for metric in ['HOTA','AssA']:
        expected = float(next(r['DiagonalMean'] for r in means if r['run']==group and r['metric']==metric))
        assert abs(statistics.mean(c[metric] for c in cells)-expected)<1e-12
    lines.append('| %s | %s | %s |' % (label,cells[0]['seed'],
                 ' | '.join('%.6f / %.6f'%(c['HOTA'],c['AssA']) for c in cells)))
write_csv(out / 'diagonal_by_domain.csv',diagonal)
lines += ['', 'Full precision: diagonal_by_domain.csv. All means match the previously reported diagonal means. '
          'The new C-series separately uses the X3 all-frame evaluator.', '']
(out / 'DIAGONAL_BY_DOMAIN.md').write_text('\n'.join(lines))
if all((Q / 'runs/C1' / ('v2_end_rank%d.json'%rank)).exists() for rank in range(4)):
    per_rank,all_rows = [],[]
    for rank in range(4):
        p = Q / 'runs/C1' / ('iterations_rank%d.csv'%rank)
        values = list(csv.DictReader(p.open()))
        end = json.loads((p.parent / ('v2_end_rank%d.json'%rank)).read_text())
        assert len(values)==228
        key = sum(r['key_trim_effective']=='True' for r in values)
        ref = sum(r['ref_trim_effective']=='True' for r in values)
        assert key==end['key_effective_batches'] and ref==end['ref_effective_batches']
        per_rank.append(dict(rank=rank,batches=228,key_effective=key,ref_effective=ref,
             key_ratio=key/228,ref_ratio=ref/228,first_initialized_iteration=end['first_initialized_iteration'],
             iterations_sha256=sha(p)))
        all_rows.extend(values)
    key = sum(r['key_trim_effective']=='True' for r in all_rows)
    ref = sum(r['ref_trim_effective']=='True' for r in all_rows)
    dump(Q / 'manifests/C1_activity_summary.json',dict(group='C1',
         definition='trim output differs from input; all local-rank batches counted',
         total_rank_batches=912,steps_per_rank=228,key_effective_batches=key,ref_effective_batches=ref,
         key_ratio=key/912,ref_ratio=ref/912,
         first_initialized_iteration_by_rank=[r['first_initialized_iteration'] for r in per_rank],by_rank=per_rank))
print('DIAGONAL_DOMAIN_TABLE_COMPLETE',len(diagonal),flush=True)
