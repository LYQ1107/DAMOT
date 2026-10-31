# ICS advantage: paired-video quantitative and qualitative evaluation

**2026-10-31 update:** The requested stage1 supplement is complete; the stage4 main comparison remains unavailable. Held-out Overcast AssA Δ is +1.582 pp [-0.240, 3.853] (8 videos); pooled Night/Dawn transfer IR Δ is +0.012122 [0.010594, 0.013722] (higher IR is worse). These one-seed results do not establish a consistent advantage. [Stage1 supplement and source-stop details](#batch2-status).

Run: `/data1/liuyeqiang/ics_advantage_20261031_004629`. Inference and offline calculations only. **One training seed: 1790355234.** All reported confidence intervals condition on these trained checkpoints; resampling videos does not estimate variation across training seeds.

The task fixes the primary model comparisons C2−C0 and C2−C4. C1/C3 are supplementary. All test-video selections use the historical split_mask seed0. Bootstrap uses10,000 paired video resamples, seed0, pointwise percentile95% intervals. Wilcoxon tests are two-sided; p values and intervals are not adjusted for multiple comparisons.

## Protocol and explicit implementation choices

- Frozen all-frame predictions are reused. Full-pool nonlinear aggregates reproduce the original scores before any paired analysis. Formal paired statistics then use test videos only, as requested; their equal-video mean differences need not equal differences between the original full-pool aggregates.
- HOTA/AssA come from the existing TrackEval per-sequence threshold arrays, combined with TrackEval functions. IDF1/IDSW use the original evaluator’s MMTracking/MOTMetrics matching. This preserves exact original-score parity rather than changing the ID evaluator to TrackEval.
- Feature sampling inherits30 frames/video at original stride2, without a video cap. All eligible GT car boxes are kept; no box-size threshold or RoI cap. The preselected annotations are inferred at effective stride1. Dose views use the first five sampled frames. Metrics use L2-normalized raw association embeddings, without any inference-time ICS projection or shrinkage.
- IR uses the same eligible RoIs in its numerator and nearest-other-car denominator. Single-identity frames are omitted from IR only. Each dose retrieval uses its finite-view support and a matched-support clean baseline; the P2 temporal window and max4000 queries are unchanged.
- Identity labels are namespaced by video. Cosine silhouette is exact over clean/fap/photo and all available12 dose views, using cluster-mean algebra instead of a quadratic pairwise-distance matrix. It is independently checked against sklearn.
- The summary IR/mAP is an equal mean over the14 requested intervention views. All individual views, strengths and paired effects are retained.
- t-SNE’s four uniform trajectory positions can lie outside the first-five dose frames. Dedicated extractions supply those three visualization views; they are excluded from the primary quantitative metrics. Tracks, frames and background are selected from metadata before feature outcomes. Each model uses its own PCA50 and t-SNE with the requested fixed parameters.

## Batch1: C0–C4 Night epoch2

## Table1 — Paired tracking differences

OVERALL is the primary category, matching the original headline scores. HOTA/AssA/IDF1 differences are percentage points; IDSW differences are switches/video (negative is favorable). Brightness thirds are fixed by test-Night brightness, then video-name tie-break; every original frame contributes to each video’s brightness. Car and all underlying sequence results are included in the linked CSVs.

| Domain / stratum | Comparison | Metric | Videos | Mean | Median | Improved | 95% CI | Wilcoxon p |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| night / all_test_videos | C2−C0 | HOTA | 25 | 0.140 | 0.012 | 52.0% | [-0.289, 0.600] | 0.9036 |
| night / all_test_videos | C2−C0 | AssA | 25 | 0.391 | 0.119 | 52.0% | [-0.592, 1.471] | 0.6571 |
| night / all_test_videos | C2−C0 | IDF1 | 25 | 0.130 | 0.134 | 56.0% | [-0.584, 0.862] | 0.7366 |
| night / all_test_videos | C2−C0 | IDSW | 25 | -1.600 | -1.000 | 52.0% | [-10.000, 4.480] | 0.8188 |
| night / night_within_domain:darkest | C2−C0 | HOTA | 9 | 0.286 | 0.041 | 66.7% | [-0.556, 1.345] | 0.3743 |
| night / night_within_domain:darkest | C2−C0 | AssA | 9 | 0.739 | 0.119 | 55.6% | [-0.931, 2.972] | 0.3139 |
| night / night_within_domain:darkest | C2−C0 | IDF1 | 9 | 0.521 | 0.099 | 55.6% | [-0.689, 2.034] | 0.5147 |
| night / night_within_domain:darkest | C2−C0 | IDSW | 9 | 0.333 | -2.000 | 55.6% | [-6.111, 8.556] | 0.6779 |
| night / night_within_domain:middle | C2−C0 | HOTA | 8 | -0.038 | -0.227 | 37.5% | [-0.492, 0.445] | 0.6744 |
| night / night_within_domain:middle | C2−C0 | AssA | 8 | -0.054 | -0.121 | 50.0% | [-1.104, 1.056] | 1 |
| night / night_within_domain:middle | C2−C0 | IDF1 | 8 | -0.127 | -0.304 | 50.0% | [-0.863, 0.623] | 0.7794 |
| night / night_within_domain:middle | C2−C0 | IDSW | 8 | -0.875 | -2.000 | 62.5% | [-4.500, 2.500] | 0.8316 |
| night / night_within_domain:brightest | C2−C0 | HOTA | 8 | 0.152 | 0.059 | 50.0% | [-0.595, 0.974] | 0.7794 |
| night / night_within_domain:brightest | C2−C0 | AssA | 8 | 0.443 | 0.163 | 50.0% | [-1.540, 2.645] | 0.7794 |
| night / night_within_domain:brightest | C2−C0 | IDF1 | 8 | -0.052 | 0.277 | 62.5% | [-1.594, 1.344] | 0.7794 |
| night / night_within_domain:brightest | C2−C0 | IDSW | 8 | -4.500 | 2.500 | 37.5% | [-28.128, 10.750] | 0.4406 |
| dawn_dusk / all_test_videos | C2−C0 | HOTA | 17 | 0.022 | 0.033 | 52.9% | [-0.256, 0.309] | 0.9434 |
| dawn_dusk / all_test_videos | C2−C0 | AssA | 17 | 0.084 | 0.081 | 52.9% | [-0.534, 0.753] | 0.9434 |
| dawn_dusk / all_test_videos | C2−C0 | IDF1 | 17 | 0.085 | 0.017 | 52.9% | [-0.366, 0.595] | 0.8313 |
| dawn_dusk / all_test_videos | C2−C0 | IDSW | 17 | -1.118 | 0.000 | 47.1% | [-2.824, 0.412] | 0.252 |
| night / all_test_videos | C2−C4 | HOTA | 25 | 0.122 | 0.110 | 60.0% | [-0.236, 0.480] | 0.4432 |
| night / all_test_videos | C2−C4 | AssA | 25 | 0.193 | 0.130 | 56.0% | [-0.634, 1.045] | 0.6571 |
| night / all_test_videos | C2−C4 | IDF1 | 25 | 0.021 | 0.285 | 56.0% | [-0.597, 0.588] | 0.6766 |
| night / all_test_videos | C2−C4 | IDSW | 25 | -0.400 | 1.000 | 48.0% | [-4.801, 4.120] | 0.6764 |
| night / night_within_domain:darkest | C2−C4 | HOTA | 9 | 0.317 | 0.110 | 55.6% | [-0.188, 0.920] | 0.4413 |
| night / night_within_domain:darkest | C2−C4 | AssA | 9 | 0.646 | 0.089 | 55.6% | [-0.348, 1.949] | 0.5147 |
| night / night_within_domain:darkest | C2−C4 | IDF1 | 9 | 0.365 | 0.324 | 55.6% | [-0.356, 1.082] | 0.3743 |
| night / night_within_domain:darkest | C2−C4 | IDSW | 9 | -3.778 | -6.000 | 55.6% | [-11.778, 4.222] | 0.3428 |
| night / night_within_domain:middle | C2−C4 | HOTA | 8 | 0.372 | 0.236 | 62.5% | [-0.166, 1.016] | 0.4838 |
| night / night_within_domain:middle | C2−C4 | AssA | 8 | 0.964 | 0.429 | 62.5% | [-0.315, 2.537] | 0.4008 |
| night / night_within_domain:middle | C2−C4 | IDF1 | 8 | 0.243 | 0.122 | 50.0% | [-0.687, 1.218] | 0.6744 |
| night / night_within_domain:middle | C2−C4 | IDSW | 8 | -3.625 | -5.000 | 62.5% | [-7.750, 0.500] | 0.141 |
| night / night_within_domain:brightest | C2−C4 | HOTA | 8 | -0.347 | 0.081 | 62.5% | [-1.008, 0.229] | 0.6744 |
| night / night_within_domain:brightest | C2−C4 | AssA | 8 | -1.086 | -0.337 | 50.0% | [-2.582, 0.285] | 0.4838 |
| night / night_within_domain:brightest | C2−C4 | IDF1 | 8 | -0.588 | 0.175 | 62.5% | [-1.960, 0.569] | 0.7794 |
| night / night_within_domain:brightest | C2−C4 | IDSW | 8 | 6.625 | 2.000 | 25.0% | [-0.875, 14.500] | 0.1065 |
| dawn_dusk / all_test_videos | C2−C4 | HOTA | 17 | -0.082 | 0.057 | 52.9% | [-0.347, 0.155] | 0.7583 |
| dawn_dusk / all_test_videos | C2−C4 | AssA | 17 | -0.103 | 0.072 | 58.8% | [-0.722, 0.482] | 0.9811 |
| dawn_dusk / all_test_videos | C2−C4 | IDF1 | 17 | -0.033 | 0.042 | 52.9% | [-0.433, 0.370] | 0.9811 |
| dawn_dusk / all_test_videos | C2−C4 | IDSW | 17 | -1.059 | 0.000 | 29.4% | [-3.235, 0.882] | 0.7774 |

Table1 conclusion: Night AssA favors C2 with a95% interval entirely above zero in 0/2 primary comparisons. These are paired-video results for one trained seed; HOTA, IDF1, IDSW and brightness strata are all shown above.

[All categories and strata](analysis/batch1/tracking/paired_effects.csv) · [Per-video differences](analysis/batch1/tracking/video_differences.csv) · [Brightness-bin membership](analysis/batch1/tracking/brightness_bins.csv) · [Spearman correlations](analysis/batch1/tracking/spearman.csv)

| Domain | Comparison | AssA Δ vs brightness: Spearman ρ | p |
| --- | --- | --- | --- |
| night | C2−C0 | -0.06538461538461539 | 0.7561668989895343 |
| dawn_dusk | C2−C0 | -0.14460784313725492 | 0.5797547992794121 |
| night | C2−C4 | -0.2053846153846154 | 0.3246727131254855 |
| dawn_dusk | C2−C4 | -0.1593137254901961 | 0.541361632626163 |

Expected environmental pattern: negative AssA correlation (larger gains for darker videos). The plot shows all test-Night videos and the predeclared third-wise mean differences with video-bootstrap intervals.

## Table2 — Intervention invariance

IR is lower-is-better; mAP and cosine silhouette are higher-is-better. IR/mAP intervals resample videos jointly across models and views. Silhouette is the exact mixed-view point estimate; no silhouette confidence interval was requested. The table summarizes all14 intervention views; the unabridged per-view table is linked below.

| Pool | Model | Mean IR [95% CI] | Mean cross-intervention mAP% [95% CI] | Cosine silhouette |
| --- | --- | --- | --- | --- |
| night | C0 | 0.3862 [0.3549, 0.4166] | 77.394 [69.445, 84.889] | 0.04279 |
| dawn_dusk | C0 | 0.2877 [0.2585, 0.3141] | 76.123 [66.288, 85.919] | 0.01961 |
| pooled | C0 | 0.3508 [0.3228, 0.3789] | 76.810 [70.585, 82.902] | 0.00366 |
| night | C1 | 0.3861 [0.3549, 0.4164] | 77.456 [69.578, 84.919] | 0.04267 |
| dawn_dusk | C1 | 0.2890 [0.2595, 0.3156] | 76.042 [66.339, 85.704] | 0.01834 |
| pooled | C1 | 0.3514 [0.3235, 0.3793] | 76.819 [70.648, 82.858] | 0.00333 |
| night | C2 | 0.3871 [0.3556, 0.4179] | 77.210 [69.238, 84.776] | 0.04233 |
| dawn_dusk | C2 | 0.2882 [0.2591, 0.3146] | 76.063 [66.457, 85.628] | 0.01858 |
| pooled | C2 | 0.3517 [0.3235, 0.3800] | 76.664 [70.437, 82.738] | 0.00304 |
| night | C3 | 0.3869 [0.3554, 0.4177] | 77.294 [69.396, 84.803] | 0.04260 |
| dawn_dusk | C3 | 0.2881 [0.2590, 0.3143] | 76.215 [66.503, 85.842] | 0.02076 |
| pooled | C3 | 0.3515 [0.3232, 0.3797] | 76.774 [70.605, 82.845] | 0.00381 |
| night | C4 | 0.3870 [0.3553, 0.4179] | 77.304 [69.318, 84.837] | 0.04049 |
| dawn_dusk | C4 | 0.2878 [0.2587, 0.3143] | 76.004 [66.191, 85.846] | 0.01863 |
| pooled | C4 | 0.3514 [0.3229, 0.3798] | 76.712 [70.459, 82.825] | 0.00209 |

| Pool | Comparison | Metric | Paired Δ | 95% CI | CI entirely favorable |
| --- | --- | --- | --- | --- | --- |
| night | C2−C0 | IR | +0.00087 | [0.000053, 0.001780] | False |
| night | C2−C0 | mAP | -0.18440 | [-0.440, 0.045] | False |
| dawn_dusk | C2−C0 | IR | +0.00048 | [-0.000086, 0.001170] | False |
| dawn_dusk | C2−C0 | mAP | -0.06044 | [-0.406, 0.225] | False |
| pooled | C2−C0 | IR | +0.00085 | [0.000278, 0.001488] | False |
| pooled | C2−C0 | mAP | -0.14651 | [-0.351, 0.038] | False |
| night | C2−C4 | IR | +0.00008 | [-0.000554, 0.000755] | False |
| night | C2−C4 | mAP | -0.09438 | [-0.345, 0.110] | False |
| dawn_dusk | C2−C4 | IR | +0.00042 | [-0.000258, 0.001012] | False |
| dawn_dusk | C2−C4 | mAP | +0.05844 | [-0.308, 0.333] | False |
| pooled | C2−C4 | IR | +0.00031 | [-0.000186, 0.000812] | False |
| pooled | C2−C4 | mAP | -0.04871 | [-0.246, 0.121] | False |

Table2 conclusion: pooled mean IR supports C2 over 0/2 controls with a wholly favorable95% interval; pooled mean cross-intervention mAP supports C2 over 0/2 controls. Individual strengths can differ; all are retained, and no favorable subset is selected.

[Every model/view/strength, including clean baselines](analysis/batch1/features/per_view.csv) · [All paired model effects](analysis/batch1/features/paired_effects.csv) · [Exact silhouette sample counts](analysis/batch1/features/silhouette.csv) · [Fixed P2 query supports](analysis/batch1/features/query_supports.json)

[Unabridged per-view Markdown table](analysis/batch1/features/PER_VIEW_TABLE.md)

## Figures

[Brightness–AssA paired effects](analysis/batch1/tracking/brightness_AssA.png) · [PDF](analysis/batch1/tracking/brightness_AssA.pdf)

[Pooled dose curves](figures/batch1/dose_response_pooled.png) · [PDF](figures/batch1/dose_response_pooled.pdf)

[Pooled IR/mAP/silhouette bars](figures/batch1/invariance_bars_pooled.png) · [PDF](figures/batch1/invariance_bars_pooled.pdf)

[Intervention t-SNE: C0 / C4 / C2](figures/batch1/intervention_tsne.png) · [PDF](figures/batch1/intervention_tsne.pdf)

Night and Dawn/Dusk dose curves and bars are also included. t-SNE axes are independently fitted per model; their coordinate ranges and apparent clusters are qualitative, and do not replace the quantitative tables.

## Batch2 status

Source training task: `/data1/liuyeqiang/damot_hf_intics_20261031_004258`. Stage4 epoch5 is the main comparison; stage1 epoch5 Overcast is supplementary. For both checkpoints the feature protocol remains the requested Night/Dawn-Dusk test pool, so the stage1 feature results measure transfer to those domains.

Controller state: `waiting_source_training_and_evaluation`.

Source task state: `formal_training_or_idle_GPU_queue`.

**Stage4 is not yet available; the requested stage1 supplement is complete.**

Historical stage1 stop record captured when this supplement was dispatched: `{"status": "stop", "reasons": ["bank PC1/offline FAP PC1 angle exceeds45deg"], "HOTA": 36.47477411631231, "HF_R11_HOTA": 36.781974908118556, "HOTA_delta": -0.3072007918062454, "first_direction_angle_deg": 82.48092299544699, "first_direction_to_offline_span_deg": 33.75608908725687, "principal_angles_deg": [33.453465219188544, 88.357491312671], "P1_weather_shift_energy_share": 0.08507769566556955, "P1_identity_energy_share": 0.09112870290405677, "seed": 1790355234, "time": 1793395076.516372}`.

Source status update (2026-10-31T16:02:37.954788+08:00): the source task records an authorized correction of its gate-angle interpretation to `first_direction_to_offline_span_deg` = 33.7561°, gate status `pass`; its controller state is `formal_training_or_idle_GPU_queue`. The historical axis-to-axis angle above and this direction-to-subspace angle are different quantities.

Frozen full-pool Overcast HOTA: INT 36.4748, HF 36.7820, Δ-0.3072 pp. Table1 below instead uses held-out test videos with an equal-video mean.

The full-run comparison jointly changes the instructed ICS estimator, shrinkage/loss settings and sample gate. APR complement coupling also follows that bank by design; this is the effect of the requested complete ICS configuration within that coupling, rather than an isolated change of estimator alone. One matched training seed is used.

## Batch2 Table1 — Paired tracking differences

| Stage / domain / stratum | Metric | Videos | Mean | Median | Improved | 95% CI | Wilcoxon p |
| --- | --- | --- | --- | --- | --- | --- | --- |
| batch2_stage1 / overcast / all_test_videos | HOTA | 8 | 0.134 | -0.055 | 37.5% | [-0.522, 0.820] | 1 |
| batch2_stage1 / overcast / all_test_videos | AssA | 8 | 1.582 | 0.492 | 62.5% | [-0.240, 3.853] | 0.327 |
| batch2_stage1 / overcast / all_test_videos | IDF1 | 8 | -0.034 | -0.218 | 37.5% | [-1.012, 0.918] | 0.8886 |
| batch2_stage1 / overcast / all_test_videos | IDSW | 8 | -4.250 | -2.000 | 62.5% | [-13.375, 2.375] | 0.4817 |

Batch2 Table1 conclusion: mean test-video AssA has a wholly positive95% interval in 0/1 reported stage/domain comparisons. All four tracking metrics and all fixed brightness strata are retained; these intervals do not measure training-seed variability.

[All categories/strata](analysis/batch2/tracking/paired_effects.csv) · [Per-video differences](analysis/batch2/tracking/video_differences.csv) · [Spearman correlations](analysis/batch2/tracking/spearman.csv) · [Brightness-bin membership CSV](analysis/batch2/tracking/brightness_bins.csv)

## Batch2 Table2 — Intervention invariance

Only stage1 Overcast tracking is available. Night tracking strata and pooled four-domain brightness strata are not applicable; their membership CSV contains its schema and no rows. The requested stage4 main comparison remains pending.

| Pool | Model | Mean IR [95% CI] | Mean cross-intervention mAP% [95% CI] | Cosine silhouette |
| --- | --- | --- | --- | --- |
| night | HF_stage1 | 0.5191 [0.4841, 0.5534] | 76.336 [68.795, 83.647] | 0.05375 |
| dawn_dusk | HF_stage1 | 0.3733 [0.3405, 0.4052] | 76.482 [66.672, 86.198] | 0.05408 |
| pooled | HF_stage1 | 0.4667 [0.4303, 0.5026] | 76.262 [70.297, 82.135] | 0.02675 |
| night | INT_stage1 | 0.5315 [0.4975, 0.5648] | 76.240 [68.634, 83.620] | 0.05073 |
| dawn_dusk | INT_stage1 | 0.3847 [0.3518, 0.4167] | 76.816 [67.022, 86.494] | 0.05610 |
| pooled | INT_stage1 | 0.4788 [0.4426, 0.5142] | 76.311 [70.305, 82.190] | 0.02646 |

| Pool | Comparison | Metric | Paired Δ | 95% CI | CI entirely favorable |
| --- | --- | --- | --- | --- | --- |
| night | INT_stage1−HF_stage1 | IR | +0.01248 | [0.010194, 0.015033] | False |
| night | INT_stage1−HF_stage1 | mAP | -0.09604 | [-0.471, 0.357] | False |
| dawn_dusk | INT_stage1−HF_stage1 | IR | +0.01148 | [0.010060, 0.012715] | False |
| dawn_dusk | INT_stage1−HF_stage1 | mAP | +0.33390 | [-0.076, 0.794] | False |
| pooled | INT_stage1−HF_stage1 | IR | +0.01212 | [0.010594, 0.013722] | False |
| pooled | INT_stage1−HF_stage1 | mAP | +0.04862 | [-0.251, 0.375] | False |

Batch2 Table2 conclusion (INT_stage1): pooled mean IR interval entirely favorable: False; pooled mean cross-intervention mAP interval entirely favorable: False. Every requested strength is reported in the per-view and paired-effect CSVs.

Pooled IR is higher (worse) for INT_stage1 across its full paired95% interval: Δ+0.012122 [0.010594, 0.013722].

[Every model/view/strength and matched clean baseline](analysis/batch2/features/per_view.csv) · [All paired feature effects](analysis/batch2/features/paired_effects.csv) · [Silhouette sample counts](analysis/batch2/features/silhouette.csv)

## Batch2 VKITTI cross-weather retrieval

Unmodified second/third-part ICS-evidence scene split and IoU pairing are reused. All1,179 test queries per variant are kept, including10 all-positive galleries with AP1. Clone temporal retrieval uses1,175 queries. Both models are BDD-trained and receive no VKITTI finetuning.

| Model | Weather | Cross-weather mAP% | Clone mAP% | Queries |
| --- | --- | --- | --- | --- |
| HF_stage1 | fog | 66.990 | 62.848 | 1179 |
| HF_stage1 | morning | 81.350 | 62.848 | 1179 |
| HF_stage1 | overcast | 81.065 | 62.848 | 1179 |
| HF_stage1 | rain | 75.868 | 62.848 | 1179 |
| HF_stage1 | sunset | 78.227 | 62.848 | 1179 |
| HF_stage1 | mean | 76.700 | 62.848 | 5895 |
| INT_stage1 | fog | 66.513 | 63.343 | 1179 |
| INT_stage1 | morning | 81.447 | 63.343 | 1179 |
| INT_stage1 | overcast | 81.458 | 63.343 | 1179 |
| INT_stage1 | rain | 75.711 | 63.343 | 1179 |
| INT_stage1 | sunset | 78.341 | 63.343 | 1179 |
| INT_stage1 | mean | 76.694 | 63.343 | 5895 |

VKITTI table conclusion (stage1): mean cross-weather mAP INT−HF is -0.005959 percentage points; the point estimate does not favor INT. This supplementary VK table reports one-seed point estimates.

[Night / stage1 Overcast brightness–AssA effects](analysis/batch2/tracking/brightness_AssA.png) · [PDF](analysis/batch2/tracking/brightness_AssA.pdf)

stage1: [Dose curves](figures/batch2/dose_response_pooled_stage1.png) · [IR/mAP/silhouette](figures/batch2/invariance_bars_pooled_stage1.png) · [Intervention t-SNE](figures/batch2/intervention_tsne_stage1.png) · [Cross-weather t-SNE](figures/batch2/cross_weather_tsne_stage1.png)

## Reproducibility and validation

Original model checkpoints, raw frozen predictions, full-resolution images and feature NPZs stay on the server. Published evidence includes all metric CSVs, statistical protocols, sample/query/trajectory selections, figures, scripts, logs and SHA256 provenance. No checkpoint, sample, view, seed, brightness bin or threshold is selected using the observed effects.
