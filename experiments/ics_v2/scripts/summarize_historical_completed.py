"""Merge twelve authorized inference cells with archived rows; uniform legacy metrics."""
from common import *
import statistics
import pickle

DOMS = ['overcast','rainy','dawn_dusk','night']
METRICS = ['HOTA','DetA','IDF1','AssA']
rows = list(csv.DictReader((Q / 'historical/stage_domain_metrics.csv').open()))
done = list((Q / 'historical_inference').glob('*/*/*/complete.json'))
assert len(done) == 12
provenance = []
for path in sorted(done):
    result = json.loads(path.read_text())
    group, stage, domain = result['group'], int(result['stage']), result['domain']
    job = json.loads((path.parent / 'job.json').read_text())
    source = Path(result['summary'])
    table = next(r for r in csv.DictReader(source.open()) if r[''] == 'OVERALL')
    track = next(r for r in csv.DictReader(source.with_name('track_metrics.csv').open()) if r[''] == 'OVERALL')
    with source.with_name('raw_results.pkl').open('rb') as f:
        raw = pickle.load(f)
    expected = {'overcast':4448, 'rainy':3150, 'dawn_dusk':6476}[domain]
    assert len(raw['det_bboxes']) == len(raw['track_bboxes']) == expected
    del raw
    original = next(r for r in rows if r['run'] == group)
    assert not any(r['run'] == group and int(r['stage']) == stage and r['domain'] == domain for r in rows)
    row = dict(run=group, method=original['method'], stage=stage, domain=domain, epoch=5,
               seed=original['seed'], protocol='archived legacy evaluator', precision='full inference CSV',
               source=str(source), **{m: float(table[m]) for m in METRICS}, IDSW=int(float(track['IDSw'])))
    rows.append(row)
    provenance.append(dict(group=group, stage=stage, domain=domain, frames=expected,
         checkpoint=job['checkpoint'], checkpoint_sha256=job['checkpoint_sha256'],
         summary_sha256=sha(source), protocol=job['protocol'], original_four_gpu_evaluator=True))
assert len(rows) == 104
out = Q / 'historical_completed'
out.mkdir(exist_ok=True)
write_csv(out / 'stage_domain_metrics.csv', rows)
dump(out / 'inference_provenance.json', provenance)
summaries = []
for group in dict.fromkeys(r['run'] for r in rows):
    data = [r for r in rows if r['run'] == group]
    for metric in METRICS:
        lookup = {(int(r['stage']),r['domain']):float(r[metric]) for r in data}
        def avg(keys):
            return statistics.mean(lookup[k] for k in keys) if all(k in lookup for k in keys) else ''
        seen = [avg([(stage,d) for d in DOMS[:stage]]) for stage in range(1,5)]
        aip = statistics.mean(seen) if all(v != '' for v in seen) else ''
        learned_complete = all((s,DOMS[j-1]) in lookup for j in range(1,4) for s in range(j,5))
        avgf = (statistics.mean(max(lookup[s,DOMS[j-1]] for s in range(j,4)) - lookup[4,DOMS[j-1]] for j in range(1,4))
                if learned_complete else '')
        bwt = (statistics.mean(lookup[4,DOMS[j-1]] - lookup[j,DOMS[j-1]] for j in range(1,4))
               if learned_complete else '')
        summaries.append(dict(run=group, metric=metric, seed=data[0]['seed'], observed_cells=len(data),
             DiagonalMean=avg([(j,d) for j,d in enumerate(DOMS,1)]),
             FinalMean=avg([(4,d) for d in DOMS]), AIP=aip, AvgF=avgf,
             FinalOld=avg([(4,d) for d in DOMS[:3]]), BWT=bwt))
write_csv(out / 'continual_metrics.csv', summaries)
comparisons = []
for treatment, control in [('SPA_oldICS','SPA_pipeline'),('F_old','SPA_APR')]:
    for metric in METRICS:
        a = next(r for r in summaries if r['run'] == treatment and r['metric'] == metric)
        b = next(r for r in summaries if r['run'] == control and r['metric'] == metric)
        assert all(a[k] != '' and b[k] != '' for k in ['FinalMean','AIP','AvgF','FinalOld','BWT','DiagonalMean'])
        comparisons.append(dict(ICS=treatment,no_ICS=control,metric=metric,ICS_seed=a['seed'],
             no_ICS_seed=b['seed'],same_seed=a['seed']==b['seed'],
             **{'delta_'+m:a[m]-b[m] for m in ['FinalMean','AIP','AvgF','FinalOld','BWT','DiagonalMean']}))
write_csv(out / 'old_ICS_comparisons.csv', comparisons)
lines = ['# Historical ICS comparison after authorized inference completion', '',
 'Twelve old-domain cells were inferred from the two existing no-ICS runs. No historical training '
 'was repeated. Original native four-GPU evaluation, Conv-BN fusion, cuDNN benchmark=True, tracker '
 'init .30 and the original legacy HOTA frame mapping are retained. New missing inference uses V100s; '
 'the archived runs/evaluations used A100s. Historical seed and APR differences remain confounders.', '',
 'All values are percentage points. AIP = equal mean of stage-wise seen-domain means. '
 'AvgF = mean over three old domains of max(R_sj, s=j..3) minus R_4j, without clipping. '
 'FinalOld = stage4 mean over three old domains. BWT = mean(R_4j-R_jj). '
 'DiagonalMean = mean(R_jj), distinct from stage4 FinalMean.', '',
 '| Run | Metric | Seed | DiagonalMean | FinalMean | AIP | AvgF | FinalOld | BWT | ΔFinalMean vs no ICS | ΔAvgF vs no ICS |',
 '|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|']
for r in summaries:
    comp = next((c for c in comparisons if c['ICS']==r['run'] and c['metric']==r['metric']),None)
    numbers = ['N/A' if r[k]=='' else '%.6f'%r[k] for k in ['DiagonalMean','FinalMean','AIP','AvgF','FinalOld','BWT']]
    deltas = ['%+.6f'%comp[k] if comp else '—' for k in ['delta_FinalMean','delta_AvgF']]
    lines.append('| %s | %s | %s | %s |' % (r['run'],r['metric'],r['seed'],' | '.join(numbers+deltas)))
lines += ['', 'Δ is old ICS minus its corresponding no-ICS run. Lower AvgF is better. '
          'Both comparisons use different historical seeds. F_old also has weak APR (.005, score .75, '
          '64 RoIs, six GPUs ×14), versus SPA_APR (.01, score .70, 128 RoIs, four GPUs ×16). '
          'The APR implementations also differ: SPA_APR uses ProtoQuasiDenseTrackHead/RoIPrototypeMemory; '
          'F_old uses CausalQuasiDenseTrackHead/StageRoIPrototypeMemory. '
          'These are descriptive comparisons; C0–C4 use a common seed and no APR.', '',
          'SPA_legacy was not requested for supplemental inference and keeps its existing diagonal '
          'results; missing cross-domain fields remain N/A. Future-domain cells were not added. '
          'All archived diagonal values are retained unchanged. These legacy results are reported '
          'separately from the all-frame HOTA used for the new C-series/X3 evaluations.', '',
          'Seed audit: ProtoCenter stage1 also has an earlier metadata-only startup log with seed884315508 '
          'and no logged train iterations. The epoch5-completed log and resolved configuration use '
          '388086182, which is the seed reported in the table. Both records are retained; auxiliary '
          'startup seeds are separated from completed-run seeds in resolved_config_audit.json.', '']
(out / 'SUMMARY.md').write_text('\n'.join(lines))
dump(Q / 'manifests/historical_completed.json', dict(status='complete',inference_cells=12,
     total_existing_plus_new_cells=104, historical_training_repeated=False, comparison_rows=8,
     all_requested_old_domain_statistics_available=True))
print('HISTORICAL_COMPARISONS_COMPLETE', flush=True)
