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
        'GPU inference uses only physical GPUs 0 and 1 under the user’s explicit instruction. '
        'Their pre-existing memory allocations are retained. Training processes on GPUs 2–9 are outside this task.' % ROOT,
        '', 'All subspaces are fitted on fit videos and metrics use held-out test videos; car only. '
        'INT is the uncentered scatter of normalized paired fap/photo/student minus normalized clean features, '
        'estimated independently for each BDD checkpoint. ICS is QR of the checkpoint bank_pcs; '
        'W and ID are label-based weather and identity oracles. R reports ten seed-0 random 2D bases. '
        'The historical video hash split is used unchanged.', '',
        '## Table 1 — Cross-view identity retrieval (②)', '',
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
    else:
        lines += ['', 'PENDING: F/HF six-weather inference and the separately identified VKITTI-trained checkpoint. '
                  'Input annotations and sample matching are complete: 9,081 GT pairs for each variant. '
                  'Every image is staged under this run to resolve the original absolute image links.']
    lines += ['', '## Table 4 — Cross-weather and clone retrieval (⑤)']
    retrieval=ROOT/'part3/retrieval.csv'
    if retrieval.exists():
        data=rows(retrieval)
        table(lines,['Model','Weather','Cross-weather mAP %','Clone mAP %','Queries'],
              [[r['model'],r['weather'],'%.3f'%float(r['cross_mAP_pp']),
                '%.3f'%float(r['clone_mAP_pp']),r['queries']] for r in data])
        gate=ROOT/'part3/gate.json'
        if gate.exists():
            g=json.loads(gate.read_text())
            lines += ['', 'Conclusion ⑤: **%s**. Full predeclared contrasts are in `part3/contrasts.csv`.'%g['status']]
    else:
        lines += ['', 'PENDING: use final Night epoch-2 checkpoints C0–C4 and X3-1 after C training completes. '
                  'Current C-series implementation uses APR off for C0–C4; X3-1 is the APR-on, no-ICS reference. '
                  'The trained C2/C3 bases were fixed using HF stage3 FAP residuals, as documented by their own manifests; '
                  'they are distinct from the stage4 fap+photo+student INT diagnostic defined for this evidence package.']
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
