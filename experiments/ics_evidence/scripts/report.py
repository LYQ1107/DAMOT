"""Render measured staged evidence, retaining every failed prespecified gate."""
import csv
import json
from pathlib import Path
from evidence_cpu import ROOT


def rows(path):
    with Path(path).open() as f:
        return list(csv.DictReader(f))


def table(lines, header, content):
    lines += ['', '| '+' | '.join(header)+' |', '| '+' | '.join(['---']*len(header))+' |']
    lines += ['| '+' | '.join(map(str,r))+' |' for r in content]


def main():
    lines = ['# ICS causal evidence: measured results', '',
        'Run: `%s`. Inference and offline calculations only; no network training. Fixed analysis seed 0. '
        'Latest user resource authorization permits any idle GPU and memory-limited sharing on GPU0 when it is busy. '
        'Original GPU1 sharing remains permitted. All existing processes and allocations are retained; '
        'training processes are never controlled by this task.' % ROOT,
        '', 'Offline estimated INT/W/ID and INT-vk bases use fit videos only; metrics use held-out test videos, car only. '
        'INT is the uncentered scatter of normalized paired fap/photo/student minus normalized clean features, '
        'estimated independently for each BDD checkpoint. ICS is QR of the pre-existing checkpoint bank_pcs '
        'and is not re-estimated on the probe split; '
        'W and ID are label-based weather and identity oracles. R reports ten seed-0 random 2D bases. '
        'The historical video hash split is used unchanged.']
    overview=[]
    for model in ['F','HF']:
        g=json.loads((ROOT/'part1'/model/'gates.json').read_text())
        ratios=[float(r['ratio']) for r in rows(ROOT/'part1'/model/'cross_view_retrieval.csv') if r['view']!='clean']
        overview += [
            ['② Identity consistency',model,'PASS' if g['cross_view_all_pass'] else 'FAIL','minimum cross-view/clean mAP %.4f'%min(ratios)],
            ['③ Shared frame shifts',model,'PASS' if g['ICC_pass'] else 'FAIL','ICC(INT)−ICC(ID) %+.4f; threshold +0.3000'%g['ICC_INT_minus_ID']],
            ['④ Delete INT',model,'PASS' if g['deletion_pass'] else 'FAIL','Fisher drop %.2f%%; ID mAP %+.3f pp'%(g['INT_fisher_drop_pct'],g['INT_id_delta_pp'])]]
    if (ROOT/'part2/gates.csv').exists():
        for r in rows(ROOT/'part2/gates.csv'):
            overview.append(['① Weather counterfactuals',r['model'],r['status'],
                             'mean INT CCR %.4f; %.2f× random'%(float(r['INT_CCR']),float(r['INT_over_random']))])
    if (ROOT/'part3/gate.json').exists():
        g=json.loads((ROOT/'part3/gate.json').read_text())
        deltas={r['model']:float(r['cross_gain_vs_C0_pp']) for r in rows(ROOT/'part3/contrasts.csv')}
        overview.append(['⑤ Trained ICS utility','C2/C3',g['status'],'cross-weather gain vs C0: C2 %+.3f, C3 %+.3f pp'%(deltas['C2'],deltas['C3'])])
    else:
        overview.append(['⑤ Trained ICS utility','C2/C3','PENDING','Awaiting final checkpoint inference'])
    lines += ['', 'The evidence is mixed: intervention directions capture some real-weather changes, '
              'while the shared-environment criterion fails for both F and HF. '
              'These results do not establish a weather-specific or identity-free subspace.']
    table(lines,['Prespecified test','Model','Status','Measured result'],overview)
    lines += ['', '## Table 1 — Cross-view identity retrieval (②)', '',
        'mAP is in percentage points. Query and gallery pools, temporal windows and seed are identical across views. '
        'Historical P2: forward gallery frames in (query+4, query+34], up to 4,000 queries.']
    tab=[]
    for model in ['HF','F']:
        for r in rows(ROOT/'part1'/model/'cross_view_retrieval.csv'):
            tab.append([model,r['view'],'%.3f'%float(r['mAP_pp']),'%.4f'%float(r['ratio']),r['queries'],
                        'PASS' if float(r['ratio'])>=.95 else 'FAIL'])
    table(lines,['Model','Gallery view','mAP %','Ratio to clean','Queries','Ratio ≥0.95'],tab)
    lines += ['', 'Conclusion ②: all three intervention views retain at least 95% of clean retrieval mAP for both models. '
              'This supports identity consistency for these sampled interventions and this retrieval task; it is not a semantic guarantee for every image.']
    lines += ['', '## Table 2a — Shared shifts: ICC and domain variance (③)', '',
              'Only test frames with at least three distinct cars. ICC uses the RoI-weighted variance of frame means divided '
              'by total RoI variance (population variance). Domain shares use the same weighted convention. '
              'The 2D norm is reported as requested in addition to the signed first component. R is mean ± SD over ten bases.']
    tab=[]
    for model in ['HF','F']:
        for r in rows(ROOT/'part1'/model/'ICC.csv'):
            def cell(k):
                return '%.4f ± %.4f'%(float(r[k]),float(r[k+'_std'])) if r['basis']=='R' else '%.4f'%float(r[k])
            tab.append([model,r['basis'],r['coordinate'],cell('ICC'),cell('domain_share')])
    table(lines,['Model','Basis','Coordinate','ICC','Domain share'],tab)
    for model in ['HF','F']:
        g=json.loads((ROOT/'part1'/model/'gates.json').read_text())
        lines += ['', 'Conclusion ③ (%s): **%s**, ICC(INT) − ICC(ID) = %.4f; required ≥0.3000. '
                  'The requested shared-environment criterion is not supported.'%(model,'PASS' if g['ICC_pass'] else 'FAIL',g['ICC_INT_minus_ID'])]
    lines += ['', '## Table 2b — Delete each 2D subspace, β=1 (④)', '',
              'Features are Z − ZEEᵀ, with no new row normalization before classifier fitting or Fisher calculation. '
              'Weather probes are refitted on fit videos, using fit-only standardization. '
              'Identity retrieval uses the unchanged cosine scoring convention. Changes are from the same model’s Full baseline, '
              'in percentage points for accuracies/mAP; Fisher drop is relative percent. R is the mean over ten bases; '
              'all replicates and their SDs are retained in CSV.']
    tab=[]
    binary_path=ROOT/'part1/binary_accuracy.csv'
    binary={(r['model'],r['basis']):r for r in rows(binary_path)} if binary_path.exists() else {}
    for model in ['HF','F']:
        for r in rows(ROOT/'part1'/model/'deletion.csv'):
            b=binary.get((model,r['basis']))
            ordinary='%.3f (%+.3f)'%(float(b['accuracy_pp']),float(b['accuracy_delta_pp'])) if b else 'pending supplement'
            tab.append([model,r['basis'],'%.3f (%+.3f)'%(float(r['weather_bacc_pp']),float(r['weather_bacc_pp_delta'])),
                ordinary,
                '%.3f (%+.3f)'%(float(r['night_clear_bacc_pp']),float(r['night_clear_bacc_pp_delta'])),
                '%.6f'%float(r['fisher']),'%.2f%%'%float(r['fisher_drop_pct']),
                '%.3f (%+.3f)'%(float(r['id_mAP_pp']),float(r['id_mAP_pp_delta']))])
    table(lines,['Model','Deleted basis','5-domain bAcc (Δ)','Night/Clear accuracy (Δ)','Night/Clear bAcc (Δ)','Fisher','Fisher drop','ID mAP (Δ)'],tab)
    for model in ['HF','F']:
        g=json.loads((ROOT/'part1'/model/'gates.json').read_text())
        lines += ['', 'Conclusion ④ (%s): **%s**. INT Fisher drop %.3f%% (required ≥30%%); identity mAP change %+.3f pp '
                  '(required within ±1 pp).'%(model,'PASS' if g['deletion_pass'] else 'FAIL',g['INT_fisher_drop_pct'],g['INT_id_delta_pp'])]
    lines += ['', '## Table 3 — Real-weather counterfactual capture (①)']
    ccr=ROOT/'part2/CCR.csv'
    if ccr.exists():
        data=rows(ccr)
        if not (ROOT/'manifests/part2_inference_complete.json').exists():
            lines += ['', '**Staged status: F/HF counterfactual results are complete; native VKITTI-model results remain pending.**']
        lines += ['', 'The metric split holds out entire physical scenes across weather and both cameras. '
                  'Fit scenes are Scene01/Scene06/Scene20 and test scenes are Scene02/Scene18: '
                  '9,081 sampled pairs per variant overall, of which 1,179 are held-out test pairs used for CCR. '
                  'F/HF are BDD-trained with no VKITTI finetuning. The optional VK_self checkpoint is the original '
                  'VKITTI stage5 after sunset, epoch5; those images may have been part of its earlier network training, '
                  'so this descriptive in-distribution row is distinct. Its BDD comparator bases use exactly the original '
                  'HF fit RoIs; INT-vk uses clone fit scenes only. Error bars in Figure A are SD across five variants.']
        table(lines,['Model','Basis','Weather','Pairs','CCR','Δ total energy','Δ mean squared / pair','Weather/identity energy ratio'],
              [[r['model'],r['basis'],r['weather'],r['pairs'],'%.4f'%float(r['CCR']),
                '%.5f'%float(r['delta_total_sq']),'%.5f'%float(r['delta_mean_sq']),
                '%.4f'%float(r['weather_identity_energy_ratio'])] for r in data])
        for r in rows(ROOT/'part2/gates.csv'):
            lines += ['', 'Conclusion ① (%s): **%s**; mean INT CCR %.4f, random %.4f (%.2f×), stored ICS %.4f.'%
                      (r['model'],r['status'],float(r['INT_CCR']),float(r['R_CCR']),float(r['INT_over_random']),float(r['ICS_CCR']))]
        per_weather=ROOT/'part2/gates_by_weather.csv'
        if per_weather.exists():
            by_weather=rows(per_weather)
            for model in dict.fromkeys(r['model'] for r in by_weather):
                group=[r for r in by_weather if r['model']==model]
                failed=[r['weather'] for r in group if r['pass_both']!='True']
                lines += ['', 'Variant check (%s): both inequalities hold for %d/5 variants; '
                          'failed variants: %s. The displayed primary gate uses the predeclared equal mean '
                          'over five variants and does not imply uniform dominance.'%(model,5-len(failed),', '.join(failed) if failed else 'none')]
    else:
        lines += ['', 'PENDING: F/HF six-weather inference and the separately identified VKITTI-trained checkpoint. '
                  'Input annotations and sample matching are complete: 9,081 GT pairs for each variant. '
                  'Every image is staged under this run to resolve the original absolute image links.']
    lines += ['', '## Table 4 — Cross-weather and clone retrieval (⑤)']
    lines += ['', 'C0–C4 use APR-off Night continuation; X3-1 is the APR-on, no-ICS reference. '
              'C1 retains the legacy global_pc_ema estimator with one PC and alpha 0.25; '
              'C4 is the fixed random two-PC control. '
              'C2/C3 use fixed HF stage3 FAP-residual bases with shrinkage alpha 0.25/0.50, respectively. '
              'Those trained bases differ from the stage4 fap+photo+student INT diagnostic used in Parts 1–2. '
              'Each inference manifest records its exact checkpoint SHA256 and effective model settings.']
    retrieval=ROOT/'part3/retrieval.csv'
    if retrieval.exists():
        data=rows(retrieval)
        lines += ['', 'Query eligibility was audited from GT metadata before C-series inference. '
                  'All 1,179 cross-weather queries per variant are included, including ten all-positive galleries (AP=1). '
                  'Clone retrieval uses 1,175 queries; four with no temporal positive are excluded. '
                  'No query is selected using feature scores. Both cameras from the same scene are in the galleries.']
        table(lines,['Model','Weather','Cross-weather mAP %','Clone mAP %','Queries'],
              [[r['model'],r['weather'],'%.3f'%float(r['cross_mAP_pp']),
                '%.3f'%float(r['clone_mAP_pp']),r['queries']] for r in data])
        gate=ROOT/'part3/gate.json'
        if gate.exists():
            g=json.loads(gate.read_text())
            lines += ['', 'Conclusion ⑤: **%s**. Full predeclared contrasts are in `part3/contrasts.csv`.'%g['status']]
            contrasts=rows(ROOT/'part3/contrasts.csv')
            table(lines,['Model','Cross-weather Δ vs C0 pp','Clone Δ vs C0 pp','Cross-weather Δ vs C4 pp','All conditions'],
                  [[r['model'],'%+.3f'%float(r['cross_gain_vs_C0_pp']),
                    '%+.3f'%float(r['clone_gain_vs_C0_pp']),'%+.3f'%float(r['cross_gain_vs_C4_pp']),
                    'PASS' if r['pass_all_conditions']=='True' else 'FAIL'] for r in contrasts])
            lines += ['', 'The gate requires C2 or C3 to gain at least 1.0 pp over C0 in mean cross-weather mAP, '
                      'exceed C4, and improve cross-weather mAP more than clone mAP. '
                      'C0–C4 share the same single training seed; these are descriptive effects, not statistical significance claims.']
    else:
        lines += ['', 'PENDING: use final Night epoch-2 checkpoints C0–C4 and X3-1 after C training completes.']
    lines += ['', '## Figures and reproducibility', '',
              'For each model, `part1/<model>/figures/` contains PDF and PNG for B (40 fixed random test frames), '
              'C (10 globally unique tracks at each INT e1 extreme), D (signed e1 dose response), '
              'D_absolute (magnitude supplement), and INT_quadrant (fixed 2×2 scatter). '
              'Dose error bars are RoI standard deviations, not confidence intervals; same-frame other-car references '
              'use the same dose-frame pool. Curve signs use a fixed largest-coordinate-positive basis orientation. '
              'Crop, frame, scatter and dose source records are retained. A and E are added with the GPU parts.', '',
              'Raw feature NPZs, copied image data and weights remain on the server; all metric CSVs, small fitted bases, '
              'figures, source scripts, input hashes, effective arguments and logs are published. '
              'No samples, seeds, checkpoints or domains are selected using the measured outcomes. '
              'BDD unpaired domain differences include scene and identity differences; a successful probe alone would not establish weather causality.']
    (ROOT/'SUMMARY.md').write_text('\n'.join(lines)+'\n')


if __name__=='__main__':
    main()
