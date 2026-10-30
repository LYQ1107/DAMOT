"""Report every comparison and uncertainty without selecting favorable views."""
from common import *

def table(lines,headers,rows):
    lines.extend(['','| '+' | '.join(headers)+' |','| '+' | '.join(['---']*len(headers))+' |'])
    lines.extend('| '+' | '.join(map(str,r))+' |' for r in rows)

def ci(r,a,b,digits=3):return ('[%.'+str(digits)+'f, %.'+str(digits)+'f]')%(float(r[a]),float(r[b]))

def append_batch2(lines):
    lines+=['','## Batch2 status','',
        'Source training task: `%s`. Stage4 epoch5 is the main comparison; stage1 epoch5 Overcast is supplementary. '
        'For both checkpoints the feature protocol remains the requested Night/Dawn-Dusk test pool, so the stage1 feature results measure transfer to those domains.'%NEW]
    status=ROOT/'manifests/batch2_status.json'
    if status.exists():lines+=['','Controller state: `'+json.loads(status.read_text())['status']+'`.']
    complete=ROOT/'manifests/batch2_analysis_complete.json'
    if not complete.exists():
        lines+=['','Pending the independently trained final checkpoints and frozen evaluations. '
                'The prepared inference/analysis queue is waiting; no second-batch measurements are claimed yet. '
                'An externally stopped source run will be reported explicitly.']
        return
    result=json.loads(complete.read_text())
    if not result['stage4_available']:
        lines+=['','**Source run stopped at its mandatory stage1 gate. Stage4 is unavailable; only the requested stage1 supplement can be measured.**',
                '', 'Source stop record: `'+json.dumps(result.get('source_stop'),ensure_ascii=False)+'`.']
    lines+=['','The full-run comparison jointly changes the instructed ICS estimator, shrinkage/loss settings and sample gate. '
            'APR complement coupling also follows that bank by design; this is the effect of the requested complete ICS configuration within that coupling, '
            'rather than an isolated change of estimator alone. One matched training seed is used.','',
            '## Batch2 Table1 — Paired tracking differences']
    rr=readcsv(ROOT/'analysis/batch2/tracking/paired_effects.csv')
    table(lines,['Stage / domain / stratum','Metric','Videos','Mean','Median','Improved','95% CI','Wilcoxon p'],
        [[r['case']+' / '+r['domain']+' / '+r['stratum'],r['metric'],r['videos'],'%.3f'%float(r['mean']),
          '%.3f'%float(r['median']),'%.1f%%'%(100*float(r['improved_fraction'])),ci(r,'ci_low','ci_high'),
          '%.4g'%float(r['wilcoxon_p_two_sided'])] for r in rr if r['category']=='OVERALL'])
    primary=[r for r in rr if r['category']=='OVERALL' and r['stratum']=='all_test_videos' and r['metric']=='AssA']
    favorable=sum(float(r['ci_low'])>0 for r in primary)
    lines+=['','Batch2 Table1 conclusion: mean test-video AssA has a wholly positive95%% interval in %d/%d reported stage/domain comparisons. '
            'All four tracking metrics and all fixed brightness strata are retained; these intervals do not measure training-seed variability.'%(favorable,len(primary)),
            '', '[All categories/strata](analysis/batch2/tracking/paired_effects.csv) · '
            '[Per-video differences](analysis/batch2/tracking/video_differences.csv) · '
            '[Spearman correlations](analysis/batch2/tracking/spearman.csv) · '
            '[Pooled-four-domain brightness bins](analysis/batch2/tracking/brightness_bins.csv)',
            '', '## Batch2 Table2 — Intervention invariance']
    means=readcsv(ROOT/'analysis/batch2/features/model_means.csv')
    table(lines,['Pool','Model','Mean IR [95% CI]','Mean cross-intervention mAP% [95% CI]','Cosine silhouette'],
        [[r['pool'],r['model'],'%.4f %s'%(float(r['mean_IR']),ci(r,'IR_ci_low','IR_ci_high',4)),
          '%.3f %s'%(float(r['mean_cross_mAP']),ci(r,'mAP_ci_low','mAP_ci_high')),'%.5f'%float(r['silhouette'])] for r in means])
    effects=[r for r in readcsv(ROOT/'analysis/batch2/features/paired_effects.csv') if r['view']=='mean']
    table(lines,['Pool','Comparison','Metric','Paired Δ','95% CI','CI entirely favorable'],
        [[r['pool'],r['model']+'−'+r['reference'],r['metric'],'%+.5f'%float(r['delta']),ci(r,'ci_low','ci_high',6 if r['metric']=='IR' else 3),
          r['CI_excludes_zero_in_favorable_direction']] for r in effects])
    for model in dict.fromkeys(r['model'] for r in effects):
        group=[r for r in effects if r['model']==model and r['pool']=='pooled']
        flags={r['metric']:r['CI_excludes_zero_in_favorable_direction'] for r in group}
        lines+=['','Batch2 Table2 conclusion (%s): pooled mean IR interval entirely favorable: %s; '
                'pooled mean cross-intervention mAP interval entirely favorable: %s. '
                'Every requested strength is reported in the per-view and paired-effect CSVs.'%(model,flags['IR'],flags['mAP'])]
    lines+=['','[Every model/view/strength and matched clean baseline](analysis/batch2/features/per_view.csv) · '
            '[All paired feature effects](analysis/batch2/features/paired_effects.csv) · '
            '[Silhouette sample counts](analysis/batch2/features/silhouette.csv)',
            '', '## Batch2 VKITTI cross-weather retrieval','',
            'Unmodified second/third-part ICS-evidence scene split and IoU pairing are reused. '
            'All1,179 test queries per variant are kept, including10 all-positive galleries with AP1. '
            'Clone temporal retrieval uses1,175 queries. Both models are BDD-trained and receive no VKITTI finetuning.']
    vk=readcsv(ROOT/'analysis/batch2/vkitti/retrieval.csv')
    table(lines,['Model','Weather','Cross-weather mAP%','Clone mAP%','Queries'],
        [[r['model'],r['weather'],'%.3f'%float(r['cross_weather_mAP']),'%.3f'%float(r['clone_mAP']),r['queries']] for r in vk])
    lines+=['','[Night / stage1 Overcast brightness–AssA effects](analysis/batch2/tracking/brightness_AssA.png) · '
            '[PDF](analysis/batch2/tracking/brightness_AssA.pdf)']
    if result['stage4_available']:
        lines+=['','[All four domains pooled: brightness–AssA effects](analysis/batch2/tracking/brightness_AssA_four_domains.png) · '
                '[PDF](analysis/batch2/tracking/brightness_AssA_four_domains.pdf)']
    for stage in ['stage4','stage1']:
        if not any(r['model'].endswith(stage) for r in means):continue
        lines+=['','%s: [Dose curves](figures/batch2/dose_response_pooled_%s.png) · '
                '[IR/mAP/silhouette](figures/batch2/invariance_bars_pooled_%s.png) · '
                '[Intervention t-SNE](figures/batch2/intervention_tsne_%s.png) · '
                '[Cross-weather t-SNE](figures/batch2/cross_weather_tsne_%s.png)'%(stage,stage,stage,stage,stage)]

def main():
    lines=['# ICS advantage: paired-video quantitative and qualitative evaluation','',
        'Run: `%s`. Inference and offline calculations only. **One training seed: 1790355234.** '
        'All reported confidence intervals condition on these trained checkpoints; resampling videos does not estimate variation across training seeds.'%ROOT,'',
        'The task fixes the primary model comparisons C2−C0 and C2−C4. C1/C3 are supplementary. '
        'All test-video selections use the historical split_mask seed0. Bootstrap uses10,000 paired video resamples, seed0, '
        'pointwise percentile95% intervals. Wilcoxon tests are two-sided; p values and intervals are not adjusted for multiple comparisons.','',
        '## Protocol and explicit implementation choices','',
        '- Frozen all-frame predictions are reused. Full-pool nonlinear aggregates reproduce the original scores before any paired analysis. '
        'Formal paired statistics then use test videos only, as requested; their equal-video mean differences need not equal differences between the original full-pool aggregates.',
        '- HOTA/AssA come from the existing TrackEval per-sequence threshold arrays, combined with TrackEval functions. '
        'IDF1/IDSW use the original evaluator’s MMTracking/MOTMetrics matching. This preserves exact original-score parity rather than changing the ID evaluator to TrackEval.',
        '- Feature sampling inherits30 frames/video at original stride2, without a video cap. All eligible GT car boxes are kept; no box-size threshold or RoI cap. '
        'The preselected annotations are inferred at effective stride1. Dose views use the first five sampled frames. '
        'Metrics use L2-normalized raw association embeddings, without any inference-time ICS projection or shrinkage.',
        '- IR uses the same eligible RoIs in its numerator and nearest-other-car denominator. Single-identity frames are omitted from IR only. '
        'Each dose retrieval uses its finite-view support and a matched-support clean baseline; the P2 temporal window and max4000 queries are unchanged.',
        '- Identity labels are namespaced by video. Cosine silhouette is exact over clean/fap/photo and all available12 dose views, '
        'using cluster-mean algebra instead of a quadratic pairwise-distance matrix. It is independently checked against sklearn.',
        '- The summary IR/mAP is an equal mean over the14 requested intervention views. All individual views, strengths and paired effects are retained.',
        '- t-SNE’s four uniform trajectory positions can lie outside the first-five dose frames. Dedicated extractions supply those three visualization views; '
        'they are excluded from the primary quantitative metrics. Tracks, frames and background are selected from metadata before feature outcomes. '
        'Each model uses its own PCA50 and t-SNE with the requested fixed parameters.','',
        '## Batch1: C0–C4 Night epoch2']
    tracking=ROOT/'analysis/batch1/tracking/paired_effects.csv'
    if tracking.exists():
        a=readcsv(tracking)
        lines+=['','## Table1 — Paired tracking differences','',
                'OVERALL is the primary category, matching the original headline scores. HOTA/AssA/IDF1 differences are percentage points; '
                'IDSW differences are switches/video (negative is favorable). Brightness thirds are fixed by test-Night brightness, then video-name tie-break; '
                'every original frame contributes to each video’s brightness. Car and all underlying sequence results are included in the linked CSVs.']
        table(lines,['Domain / stratum','Comparison','Metric','Videos','Mean','Median','Improved','95% CI','Wilcoxon p'],
            [[r['domain']+' / '+r['stratum'],r['model']+'−'+r['reference'],r['metric'],r['videos'],
              '%.3f'%float(r['mean']),'%.3f'%float(r['median']),'%.1f%%'%(100*float(r['improved_fraction'])),
              ci(r,'ci_low','ci_high'),'%.4g'%float(r['wilcoxon_p_two_sided'])] for r in a if r['category']=='OVERALL'])
        primary=[r for r in a if r['category']=='OVERALL' and r['domain']=='night' and r['stratum']=='all_test_videos' and r['metric']=='AssA']
        support=sum(float(r['ci_low'])>0 for r in primary)
        lines+=['','Table1 conclusion: Night AssA favors C2 with a95%% interval entirely above zero in %d/2 primary comparisons. '
                'These are paired-video results for one trained seed; HOTA, IDF1, IDSW and brightness strata are all shown above.'%support,
                '', '[All categories and strata](analysis/batch1/tracking/paired_effects.csv) · '
                '[Per-video differences](analysis/batch1/tracking/video_differences.csv) · '
                '[Brightness-bin membership](analysis/batch1/tracking/brightness_bins.csv) · '
                '[Spearman correlations](analysis/batch1/tracking/spearman.csv)']
        ss=[r for r in readcsv(ROOT/'analysis/batch1/tracking/spearman.csv') if r['category']=='OVERALL' and r['metric']=='AssA']
        table(lines,['Domain','Comparison','AssA Δ vs brightness: Spearman ρ','p'],
             [[r['domain'],r['model']+'−'+r['reference'],r['rho'],r['p']] for r in ss])
        lines+=['','Expected environmental pattern: negative AssA correlation (larger gains for darker videos). '
                'The plot shows all test-Night videos and the predeclared third-wise mean differences with video-bootstrap intervals.']
    else:lines+=['','Table1 pending: all sequence-aggregate parity checks and all-frame brightness must finish before paired statistical reporting.']
    features=ROOT/'analysis/batch1/features/model_means.csv'
    if features.exists():
        a=readcsv(features);effects=readcsv(ROOT/'analysis/batch1/features/paired_effects.csv')
        lines+=['','## Table2 — Intervention invariance','',
              'IR is lower-is-better; mAP and cosine silhouette are higher-is-better. IR/mAP intervals resample videos jointly across models and views. '
              'Silhouette is the exact mixed-view point estimate; no silhouette confidence interval was requested. '
              'The table summarizes all14 intervention views; the unabridged per-view table is linked below.']
        table(lines,['Pool','Model','Mean IR [95% CI]','Mean cross-intervention mAP% [95% CI]','Cosine silhouette'],
            [[r['pool'],r['model'],'%.4f %s'%(float(r['mean_IR']),ci(r,'IR_ci_low','IR_ci_high',4)),
              '%.3f %s'%(float(r['mean_cross_mAP']),ci(r,'mAP_ci_low','mAP_ci_high')),'%.5f'%float(r['silhouette'])] for r in a])
        selected=[r for r in effects if r['model']=='C2' and r['view']=='mean']
        table(lines,['Pool','Comparison','Metric','Paired Δ','95% CI','CI entirely favorable'],
             [[r['pool'],r['model']+'−'+r['reference'],r['metric'],'%+.5f'%float(r['delta']),ci(r,'ci_low','ci_high',6 if r['metric']=='IR' else 3),
               r['CI_excludes_zero_in_favorable_direction']] for r in selected])
        ir=sum(r['CI_excludes_zero_in_favorable_direction']=='True' for r in selected if r['pool']=='pooled' and r['metric']=='IR')
        mp=sum(r['CI_excludes_zero_in_favorable_direction']=='True' for r in selected if r['pool']=='pooled' and r['metric']=='mAP')
        lines+=['','Table2 conclusion: pooled mean IR supports C2 over %d/2 controls with a wholly favorable95%% interval; '
                'pooled mean cross-intervention mAP supports C2 over %d/2 controls. Individual strengths can differ; '
                'all are retained, and no favorable subset is selected.'%(ir,mp),'',
                '[Every model/view/strength, including clean baselines](analysis/batch1/features/per_view.csv) · '
                '[All paired model effects](analysis/batch1/features/paired_effects.csv) · '
                '[Exact silhouette sample counts](analysis/batch1/features/silhouette.csv) · '
                '[Fixed P2 query supports](analysis/batch1/features/query_supports.json)']
        appendix=['# All first-batch feature views','',
                  'IR and mAP confidence intervals are pointwise video-bootstrap95%. Clean-reference mAP uses the same support as each intervention view.']
        table(appendix,['Pool','Model','View','IR [95% CI]','Cross mAP% [95% CI]','Matched clean mAP%','Queries','Query videos'],
            [[r['pool'],r['model'],r['view'],'%.4f %s'%(float(r['IR']),ci(r,'IR_ci_low','IR_ci_high',4)),
              '%.3f %s'%(float(r['cross_mAP']),ci(r,'cross_mAP_ci_low','cross_mAP_ci_high')),
              '%.3f'%float(r['clean_support_mAP']),r['queries'],r['query_videos']] for r in readcsv(ROOT/'analysis/batch1/features/per_view.csv')])
        (ROOT/'analysis/batch1/features/PER_VIEW_TABLE.md').write_text('\n'.join(appendix)+'\n')
        lines+=['','[Unabridged per-view Markdown table](analysis/batch1/features/PER_VIEW_TABLE.md)']
    else:lines+=['','Table2 pending: all five models must have identical metadata and finite-view supports before paired comparisons.']
    lines+=['','## Figures','',
            '[Brightness–AssA paired effects](analysis/batch1/tracking/brightness_AssA.png) · '
            '[PDF](analysis/batch1/tracking/brightness_AssA.pdf)','',
            '[Pooled dose curves](figures/batch1/dose_response_pooled.png) · [PDF](figures/batch1/dose_response_pooled.pdf)','',
            '[Pooled IR/mAP/silhouette bars](figures/batch1/invariance_bars_pooled.png) · [PDF](figures/batch1/invariance_bars_pooled.pdf)','',
            '[Intervention t-SNE: C0 / C4 / C2](figures/batch1/intervention_tsne.png) · [PDF](figures/batch1/intervention_tsne.pdf)','',
            'Night and Dawn/Dusk dose curves and bars are also included. t-SNE axes are independently fitted per model; '
            'their coordinate ranges and apparent clusters are qualitative, and do not replace the quantitative tables.']
    append_batch2(lines)
    lines+=['',
            '## Reproducibility and validation','',
            'Original model checkpoints, raw frozen predictions, full-resolution images and feature NPZs stay on the server. '
            'Published evidence includes all metric CSVs, statistical protocols, sample/query/trajectory selections, figures, scripts, logs and SHA256 provenance. '
            'No checkpoint, sample, view, seed, brightness bin or threshold is selected using the observed effects.']
    (ROOT/'SUMMARY.md').write_text('\n'.join(lines)+'\n')

if __name__=='__main__':main()
