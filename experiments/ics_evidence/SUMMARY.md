# ICS causal evidence: measured results

Run: `/data1/liuyeqiang/ics_evidence_20261030_215311`. Inference and offline calculations only; no network training. Fixed analysis seed 0. GPU inference uses only physical GPUs 0 and 1 under the user’s explicit instruction. Their pre-existing memory allocations are retained. Training processes on GPUs 2–9 are outside this task.

All subspaces are fitted on fit videos and metrics use held-out test videos; car only. INT is the uncentered scatter of normalized paired fap/photo/student minus normalized clean features, estimated independently for each BDD checkpoint. ICS is QR of the checkpoint bank_pcs; W and ID are label-based weather and identity oracles. R reports ten seed-0 random 2D bases. The historical video hash split is used unchanged.

## Table 1 — Cross-view identity retrieval (②)

mAP is in percentage points. Query and gallery pools, temporal windows and seed are identical across views. Historical P2: forward gallery frames in (query+4, query+34], up to 4,000 queries.

| Model | Gallery view | mAP % | Ratio to clean | Queries | Ratio ≥0.95 |
| --- | --- | --- | --- | --- | --- |
| HF | clean | 57.489 | 1.0000 | 4000 | PASS |
| HF | fap | 56.687 | 0.9861 | 4000 | PASS |
| HF | photo | 57.335 | 0.9973 | 4000 | PASS |
| HF | student | 56.623 | 0.9849 | 4000 | PASS |
| F | clean | 57.314 | 1.0000 | 4000 | PASS |
| F | fap | 56.417 | 0.9843 | 4000 | PASS |
| F | photo | 57.076 | 0.9959 | 4000 | PASS |
| F | student | 56.277 | 0.9819 | 4000 | PASS |

Conclusion ②: all three intervention views retain at least 95% of clean retrieval mAP for both models. This supports identity consistency for these sampled interventions and this retrieval task; it is not a semantic guarantee for every image.

## Table 2a — Shared shifts: ICC and domain variance (③)

Only test frames with at least three distinct cars. ICC uses the RoI-weighted variance of frame means divided by total RoI variance (population variance). Domain shares use the same weighted convention. The 2D norm is reported as requested in addition to the signed first component. R is mean ± SD over ten bases.

| Model | Basis | Coordinate | ICC | Domain share |
| --- | --- | --- | --- | --- |
| HF | INT | e1 | 0.1446 | 0.0105 |
| HF | INT | norm2 | 0.1397 | 0.0219 |
| HF | ICS | e1 | 0.1565 | 0.0069 |
| HF | ICS | norm2 | 0.1491 | 0.0032 |
| HF | W | e1 | 0.3826 | 0.2364 |
| HF | W | norm2 | 0.2042 | 0.0277 |
| HF | ID | e1 | 0.1650 | 0.0058 |
| HF | ID | norm2 | 0.1366 | 0.0125 |
| HF | R | e1 | 0.1499 ± 0.0157 | 0.0085 ± 0.0050 |
| HF | R | norm2 | 0.1401 ± 0.0215 | 0.0093 ± 0.0122 |
| F | INT | e1 | 0.1393 | 0.0095 |
| F | INT | norm2 | 0.1592 | 0.0238 |
| F | ICS | e1 | 0.1652 | 0.0069 |
| F | ICS | norm2 | 0.1451 | 0.0060 |
| F | W | e1 | 0.4275 | 0.2752 |
| F | W | norm2 | 0.2348 | 0.0485 |
| F | ID | e1 | 0.1646 | 0.0061 |
| F | ID | norm2 | 0.1693 | 0.0229 |
| F | R | e1 | 0.1512 ± 0.0146 | 0.0079 ± 0.0048 |
| F | R | norm2 | 0.1365 ± 0.0158 | 0.0070 ± 0.0066 |

Conclusion ③ (HF): **FAIL**, ICC(INT) − ICC(ID) = -0.0204; required ≥0.3000. The requested shared-environment criterion is not supported.

Conclusion ③ (F): **FAIL**, ICC(INT) − ICC(ID) = -0.0252; required ≥0.3000. The requested shared-environment criterion is not supported.

## Table 2b — Delete each 2D subspace, β=1 (④)

Features are Z − ZEEᵀ, with no new row normalization before classifier fitting or Fisher calculation. Weather probes are refitted on fit videos, using fit-only standardization. Identity retrieval uses the unchanged cosine scoring convention. Changes are from the same model’s Full baseline, in percentage points for accuracies/mAP; Fisher drop is relative percent. R is the mean over ten bases; all replicates and their SDs are retained in CSV.

| Model | Deleted basis | 5-domain bAcc (Δ) | Night/Clear bAcc (Δ) | Fisher | Fisher drop | ID mAP (Δ) |
| --- | --- | --- | --- | --- | --- | --- |
| HF | Full | 36.358 (+0.000) | 83.927 (+0.000) | 0.015310 | 0.00% | 57.489 (+0.000) |
| HF | INT | 34.257 (-2.101) | 78.804 (-5.123) | 0.010017 | 34.57% | 57.510 (+0.022) |
| HF | ICS | 36.500 (+0.143) | 83.695 (-0.232) | 0.018033 | -17.79% | 58.172 (+0.683) |
| HF | W | 23.688 (-12.670) | 54.784 (-29.143) | 0.005589 | 63.50% | 57.753 (+0.265) |
| HF | ID | 36.188 (-0.170) | 84.052 (+0.125) | 0.019304 | -26.09% | 58.074 (+0.585) |
| HF | R | 36.355 (-0.003) | 83.871 (-0.056) | 0.015288 | 0.14% | 57.471 (-0.018) |
| F | Full | 34.513 (+0.000) | 82.605 (+0.000) | 0.014334 | 0.00% | 57.314 (+0.000) |
| F | INT | 32.782 (-1.730) | 77.984 (-4.622) | 0.010350 | 27.79% | 57.689 (+0.375) |
| F | ICS | 34.668 (+0.156) | 83.057 (+0.451) | 0.016550 | -15.46% | 57.730 (+0.416) |
| F | W | 22.123 (-12.389) | 54.127 (-28.478) | 0.005442 | 62.03% | 57.809 (+0.495) |
| F | ID | 34.281 (-0.231) | 82.414 (-0.191) | 0.017351 | -21.04% | 59.346 (+2.032) |
| F | R | 34.645 (+0.133) | 82.686 (+0.081) | 0.014321 | 0.09% | 57.307 (-0.007) |

Conclusion ④ (HF): **PASS**. INT Fisher drop 34.569% (required ≥30%); identity mAP change +0.022 pp (required within ±1 pp).

Conclusion ④ (F): **FAIL**. INT Fisher drop 27.795% (required ≥30%); identity mAP change +0.375 pp (required within ±1 pp).

## Table 3 — Real-weather counterfactual capture (①)

PENDING: F/HF six-weather inference and the separately identified VKITTI-trained checkpoint. Input annotations and sample matching are complete: 9,081 GT pairs for each variant. Every image is staged under this run to resolve the original absolute image links.

## Table 4 — Cross-weather and clone retrieval (⑤)

PENDING: use final Night epoch-2 checkpoints C0–C4 and X3-1 after C training completes. Current C-series implementation uses APR off for C0–C4; X3-1 is the APR-on, no-ICS reference. The trained C2/C3 bases were fixed using HF stage3 FAP residuals, as documented by their own manifests; they are distinct from the stage4 fap+photo+student INT diagnostic defined for this evidence package.

## Figures and reproducibility

For each model, `part1/<model>/figures/` contains PDF and PNG for B (40 fixed random test frames), C (10 globally unique tracks at each INT e1 extreme), D (signed e1 dose response), D_absolute (magnitude supplement), and INT_quadrant (fixed 2×2 scatter). Dose error bars are RoI standard deviations, not confidence intervals; same-frame other-car references use the same dose-frame pool. Curve signs use a fixed largest-coordinate-positive basis orientation. Crop, frame, scatter and dose source records are retained. A and E are added with the GPU parts.

Raw feature NPZs, copied image data and weights remain on the server; all metric CSVs, small fitted bases, figures, source scripts, input hashes, effective arguments and logs are published. No samples, seeds, checkpoints or domains are selected using the measured outcomes. BDD unpaired domain differences include scene and identity differences; a successful probe alone would not establish weather causality.
