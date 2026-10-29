"""Rebuild honest partial/final CSV tables and the requested report."""
from common import *
import math,statistics,fcntl
_summary_lock=(Q/'manifests/summary.lock').open('a')
fcntl.flock(_summary_lock,fcntl.LOCK_EX)
METRICS=['HOTA_percent','DetA_percent','IDF1_percent','AssA_percent']
policy_path=Q/'manifests/F_baseline_policy.json'
policy=json.loads(policy_path.read_text()) if policy_path.exists() else {}
expected_x2_rows=28 if policy.get('mode')=='skip_F_interventions' else 56
def read(path):
    with path.open() as f:return list(csv.DictReader(f))
def csv_union(path,rows):
    if not rows:return
    keys=list(dict.fromkeys(k for r in rows for k in r));write_csv(path,[{k:r.get(k,'') for k in keys} for r in rows])
grads=[];summaries=[]
for model in ['F','HF']:
    for domain in ['night','overcast']:
        base=Q/'X1'/model/domain
        if (base/'grad_compare.csv').exists():grads.extend(read(base/'grad_compare.csv'))
        if (base/'grad_summary.json').exists():
            summary=json.loads((base/'grad_summary.json').read_text())
            batches=read(base/'grad_compare.csv')
            summary['key_trim_active_batches']=sum(int(r['key_rois'])>=96 for r in batches)
            summary['param_cos_ge_099_batches']=sum(float(r['param_cos_pcl_trim'])>=.99 for r in batches if r['triggered']=='1')
            summaries.append(summary)
csv_union(Q/'X1/grad_compare.csv',grads);csv_union(Q/'X1/grad_summary.csv',summaries)
intervention=[]
for model in ['F','HF']:
    for domain in ['night','overcast']:
        base=Q/'X2'/model/domain
        zero=base/'beta0/metrics.csv'
        ref={r['category']:r for r in read(zero)} if zero.exists() else {}
        for beta in [0,1]:
            cell=base/('beta%d'%beta)
            if not (cell/'complete.json').exists():continue
            for r in read(cell/'metrics.csv'):
                for m in METRICS:r['delta_'+m]=float(r[m])-float(ref[r['category']][m])
                intervention.append(r)
csv_union(Q/'X2/intervention.csv',intervention)
x3=[]
hfref=read(HF/'results/stage_domain_metrics.csv')
old={r['category']:r for r in hfref if int(r['stage'])==3 and r['eval_domain']=='dawn_dusk'}
for root in sorted((Q/'X3').glob('X3-*')):
    if not root.is_dir():continue
    for domain in ['night','dawn_dusk']:
        cell=root/'evaluation'/domain
        if not (cell/'complete.json').exists():continue
        for r in read(cell/'metrics.csv'):
            r.update(group=root.name,seed=1790355234)
            for metric in METRICS:r['forgetting_'+metric]=float(old[r['category']][metric])-float(r[metric]) if domain=='dawn_dusk' else ''
            x3.append(r)
csv_union(Q/'X3/x3_summary.csv',x3)
audits=[]
for root in sorted((Q/'X3').glob('X3-*')):
    for path in root.glob('audit_end_rank*.json'):
        r=json.loads(path.read_text());r['group']=root.name;audits.append(r)
csv_union(Q/'X3/x3_training_audit.csv',audits)
lines=['# ICS mechanism experiments','', 'Run: `%s`'%Q,'','All metric values and differences are percentage points. Results below are fixed protocol; missing cells remain pending.','', '## Table 1 — X1 gradients','', '| Model | Domain | Loss key present | Key RoIs ≥96 | Parameter cosine, mean ± SD | Norm ratio, mean ± SD | Total vs original cosine | Key difference energy in E |','|---|---|---:|---:|---:|---:|---:|---:|']
index={(r['model'],r['domain']):r for r in summaries}
for model in ['F','HF']:
    for domain in ['night','overcast']:
        r=index.get((model,domain))
        if not r:lines.append('| %s | %s | pending | — | — | — | — | — |'%(model,domain));continue
        def fmt(k):return '%.6f ± %.6f'%(r[k+'_mean'],r[k+'_std'])
        lines.append('| %s | %s | %d/%d | %d/%d | %s | %s | %.9f | %s |'%(model,domain,r['triggered'],r['batches'],r['key_trim_active_batches'],r['batches'],fmt('param_cos_pcl_trim'),fmt('param_norm_ratio'),r['param_cos_total_pcl_mean'],fmt('key_diff_energy_in_E')))
if len(summaries)==4:
    passing=['%s/%s'%(r['model'],r['domain']) for r in summaries if r['param_cos_pcl_trim_mean']>=.99]
    failing=['%s/%s'%(r['model'],r['domain']) for r in summaries if r['param_cos_pcl_trim_mean']<.99]
    lines+=['','Interpretation under the prespecified ≥0.99 cosine rule: '+('甲 in '+', '.join(passing)+'. ' if passing else '')+('Below the 甲 threshold in '+', '.join(failing)+'. ' if failing else '')+'This is a local gradient test and does not establish equivalence of trained models.']
else:lines+=['','Interpretation: pending all four prespecified cells.']
lines+=['','Loss-key activation and a non-identity trim are different in this implementation: it still computes the additional loss when the key batch has fewer than 96 RoIs. Key trimming is then the identity, although reference trimming may remain active. All 20 batches per cell are retained; none are filtered by the measured cosine or RoI threshold.']
lines+=['','## Table 2 — X2 inference intervention','','| Model | Domain | β | HOTA | DetA | IDF1 | AssA | ΔAssA |','|---|---|---:|---:|---:|---:|---:|---:|']
for r in intervention:
    if r['category']!='OVERALL':continue
    lines.append('| %s | %s | %s | %.6f | %.6f | %.6f | %.6f | %+.6f |'%(r['model'],r['domain'],r['beta'],*[float(r[m]) for m in METRICS],float(r['delta_AssA_percent'])))
if len(intervention)==expected_x2_rows:
    vals=[r for r in intervention if r['category']=='OVERALL' and float(r['beta'])==1]
    increases=sum(float(r['delta_AssA_percent'])>0 for r in vals)
    finding=('乙 supported in every tested cell (AssA increases after removing the leading variance directions).' if increases==len(vals) else ('No tested cell supports 乙 under the prespecified AssA-increase rule.' if increases==0 else '%d/%d cells support 乙 under the AssA-increase rule.'%(increases,len(vals))))
    lines+=['','Interpretation: '+finding+' Removing the two directions completely at inference does not by itself identify the effect of soft training shrinkage at α=0.1.']
else:lines+=['','Interpretation: pending frozen baseline parity and all intervention cells.']
basis_notes=[]
for model in ['F','HF']:
    path=Q/'X2'/('basis_e_%s.json'%model)
    if path.exists():
        b=json.loads(path.read_text());basis_notes.append('%s: %.3f° and %.3f° (%d clean car embeddings)'%(model,*b['principal_angles_degrees'],b['N']))
if basis_notes:lines+=['','Principal angles between each Clear PCA basis and its checkpoint bank: '+'; '.join(basis_notes)+'.']
if policy.get('mode')=='use_current_environment':
    lines+=['','F uses the user-approved current-environment β=0 baseline for paired intervention differences. The historical baseline mismatch is retained below; it is not counted as an exact historical reproduction.']
elif policy.get('mode')=='skip_F_interventions':
    lines+=['','F interventions were not run, following the user decision to retain the exact historical-baseline gate. Completed HF cells remain independently valid.']
diags=[]
for domain in ['overcast','night']:
    diag=Q/'X2/F'/domain/'beta0_cudnn_true/baseline_parity.csv'
    if not diag.exists():diag=Q/'X2/F'/domain/'beta0/baseline_parity.csv'
    if diag.exists():diags.extend(dict(r,domain=domain) for r in read(diag))
if diags:
    lines+=['','F historical-baseline checks (using the archived cuDNN benchmark setting):','','| Domain | Metric | Archived F | Current V100 F | Difference |','|---|---|---:|---:|---:|']
    for r in diags:
        if r['category']=='OVERALL':lines.append('| %s | %s | %.9f | %.9f | %+.9f |'%(r['domain'],r['metric'],float(r['reference']),float(r['observed']),float(r['delta'])))
lines+=['','## Table 3 — X3 Night adaptation (2 epochs)','','| Group | Seed | Domain | HOTA | DetA | IDF1 | AssA | Old-domain HOTA forgetting |','|---|---:|---|---:|---:|---:|---:|---:|']
for r in x3:
    if r['category']!='OVERALL':continue
    f='%.6f'%float(r['forgetting_HOTA_percent']) if r['forgetting_HOTA_percent']!='' else '—'
    lines.append('| %s | %s | %s | %.6f | %.6f | %.6f | %.6f | %s |'%(r['group'],r['seed'],r['domain'],*[float(r[m]) for m in METRICS],f))
if len(x3)==56:
    ix={(r['group'],r['domain']):float(r['HOTA_percent']) for r in x3 if r['category']=='OVERALL'}
    outcomes=[]
    for d in ['night','dawn_dusk']:
        control=ix[('X3-0',d)];off=ix[('X3-1',d)];weight=ix[('X3-2',d)];rand=ix[('X3-3',d)]
        if abs(control-weight)<=.3:result='甲 / no difference from α=0 within 0.3 HOTA'
        elif abs(control-rand)<=.3 and control-off>.3 and rand-off>.3:result='乙'
        elif control-weight>.3 and control-rand>.3:result='丙'
        else:result='no clear difference / mixed'
        outcomes.append(d+': '+result)
    lines+=['','Interpretation under the prespecified single-seed rule: '+ '; '.join(outcomes)+'.']
else:lines+=['','Interpretation: pending all four groups. Single-seed differences within ±0.3 HOTA are treated as no difference.']
lines+=['', 'The optional second seed has not been run; the mandatory four groups take priority under shared GPU availability.','', '## Protocol details and deviations','', 'Both OVERALL and car metrics are preserved in the CSV files; the compact tables above display OVERALL. X1 records bank initialization and updates, alpha, dimension and loss activation in per-batch CSVs. X2 uses full videos and checks β=0 against the historical frozen metrics before β=1. X3 uses identical SHA256-identified archived APR history with three domains.','']
for r in json.loads((Q/'manifests/deviations.json').read_text()):lines.append('- **%s:** %s. %s'%(r['scope'],r['change'],r['reason']))
lines+=['','## Provenance','','See `inputs.txt`, `manifests/`, per-cell resolved configs, `X3/*/config_diff.json`, and `logs/`. Checkpoints and feature NPZ files stay on the server and are excluded from GitHub.','']
(Q/'SUMMARY.md').write_text('\n'.join(lines))
dump(Q/'manifests/result_status.json',dict(X1_complete_cells=len(summaries),X2_complete_cells=len(intervention)//7,X2_required_cells=expected_x2_rows//7,X3_complete_cells=len(x3)//7,F_baseline_policy=policy.get('mode','require_exact_historical_parity'),complete=len(summaries)==4 and len(intervention)==expected_x2_rows and len(x3)==56))
print('SUMMARY',len(summaries),len(intervention)//7,len(x3)//7,flush=True)
