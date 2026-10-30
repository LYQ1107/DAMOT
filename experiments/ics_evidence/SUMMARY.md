# ICS causal evidence: measured results

Run: `/data1/liuyeqiang/ics_evidence_20261030_215311`. Inference and offline calculations only; no network training. Fixed analysis seed 0. Latest user resource authorization permits any idle GPU and memory-limited sharing on GPU0 when it is busy. Original GPU1 sharing remains permitted. All existing processes and allocations are retained; training processes are never controlled by this task.

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

| Model | Deleted basis | 5-domain bAcc (Δ) | Night/Clear accuracy (Δ) | Night/Clear bAcc (Δ) | Fisher | Fisher drop | ID mAP (Δ) |
| --- | --- | --- | --- | --- | --- | --- | --- |
| HF | Full | 36.358 (+0.000) | 83.064 (+0.000) | 83.927 (+0.000) | 0.015310 | 0.00% | 57.489 (+0.000) |
| HF | INT | 34.257 (-2.101) | 77.833 (-5.231) | 78.804 (-5.123) | 0.010017 | 34.57% | 57.510 (+0.022) |
| HF | ICS | 36.500 (+0.143) | 82.872 (-0.192) | 83.695 (-0.232) | 0.018033 | -17.79% | 58.172 (+0.683) |
| HF | W | 23.688 (-12.670) | 55.565 (-27.499) | 54.784 (-29.143) | 0.005589 | 63.50% | 57.753 (+0.265) |
| HF | ID | 36.188 (-0.170) | 83.256 (+0.192) | 84.052 (+0.125) | 0.019304 | -26.09% | 58.074 (+0.585) |
| HF | R | 36.355 (-0.003) | 83.035 (-0.029) | 83.871 (-0.056) | 0.015288 | 0.14% | 57.471 (-0.018) |
| F | Full | 34.513 (+0.000) | 81.640 (+0.000) | 82.605 (+0.000) | 0.014334 | 0.00% | 57.314 (+0.000) |
| F | INT | 32.782 (-1.730) | 76.722 (-4.918) | 77.984 (-4.622) | 0.010350 | 27.79% | 57.689 (+0.375) |
| F | ICS | 34.668 (+0.156) | 82.307 (+0.667) | 83.057 (+0.451) | 0.016550 | -15.46% | 57.730 (+0.416) |
| F | W | 22.123 (-12.389) | 55.484 (-26.156) | 54.127 (-28.478) | 0.005442 | 62.03% | 57.809 (+0.495) |
| F | ID | 34.281 (-0.231) | 81.337 (-0.303) | 82.414 (-0.191) | 0.017351 | -21.04% | 59.346 (+2.032) |
| F | R | 34.645 (+0.133) | 81.726 (+0.086) | 82.686 (+0.081) | 0.014321 | 0.09% | 57.307 (-0.007) |

Conclusion ④ (HF): **PASS**. INT Fisher drop 34.569% (required ≥30%); identity mAP change +0.022 pp (required within ±1 pp).

Conclusion ④ (F): **FAIL**. INT Fisher drop 27.795% (required ≥30%); identity mAP change +0.375 pp (required within ±1 pp).

## Table 3 — Real-weather counterfactual capture (①)

The metric split holds out entire physical scenes across weather and both cameras. Fit scenes are Scene01/Scene06/Scene20 and test scenes are Scene02/Scene18: 9,081 sampled pairs per variant overall, of which 1,179 are held-out test pairs used for CCR. F/HF are BDD-trained with no VKITTI finetuning. The optional VK_self checkpoint is the original VKITTI stage5 after sunset, epoch5; those images may have been part of its earlier network training, so this descriptive in-distribution row is distinct. Its BDD comparator bases use exactly the original HF fit RoIs; INT-vk uses clone fit scenes only. Error bars in Figure A are SD across five variants.

| Model | Basis | Weather | Pairs | CCR | Δ total energy | Δ mean squared / pair | Weather/identity energy ratio |
| --- | --- | --- | --- | --- | --- | --- | --- |
| F | INT | fog | 1179 | 0.0998 | 619.66432 | 0.52558 | 0.4107 |
| F | ICS | fog | 1179 | 0.0484 | 619.66432 | 0.52558 | 0.4107 |
| F | W | fog | 1179 | 0.0300 | 619.66432 | 0.52558 | 0.4107 |
| F | ID | fog | 1179 | 0.0519 | 619.66432 | 0.52558 | 0.4107 |
| F | R | fog | 1179 | 0.0073 | 619.66432 | 0.52558 | 0.4107 |
| F | INT-vk | fog | 1179 | 0.0767 | 619.66432 | 0.52558 | 0.4107 |
| F | INT | morning | 1179 | 0.0484 | 223.06451 | 0.18920 | 0.1478 |
| F | ICS | morning | 1179 | 0.0502 | 223.06451 | 0.18920 | 0.1478 |
| F | W | morning | 1179 | 0.0423 | 223.06451 | 0.18920 | 0.1478 |
| F | ID | morning | 1179 | 0.0481 | 223.06451 | 0.18920 | 0.1478 |
| F | R | morning | 1179 | 0.0079 | 223.06451 | 0.18920 | 0.1478 |
| F | INT-vk | morning | 1179 | 0.0406 | 223.06451 | 0.18920 | 0.1478 |
| F | INT | overcast | 1179 | 0.0411 | 245.32115 | 0.20808 | 0.1626 |
| F | ICS | overcast | 1179 | 0.0387 | 245.32115 | 0.20808 | 0.1626 |
| F | W | overcast | 1179 | 0.0365 | 245.32115 | 0.20808 | 0.1626 |
| F | ID | overcast | 1179 | 0.0400 | 245.32115 | 0.20808 | 0.1626 |
| F | R | overcast | 1179 | 0.0077 | 245.32115 | 0.20808 | 0.1626 |
| F | INT-vk | overcast | 1179 | 0.0344 | 245.32115 | 0.20808 | 0.1626 |
| F | INT | rain | 1179 | 0.0488 | 383.89481 | 0.32561 | 0.2544 |
| F | ICS | rain | 1179 | 0.0469 | 383.89481 | 0.32561 | 0.2544 |
| F | W | rain | 1179 | 0.0416 | 383.89481 | 0.32561 | 0.2544 |
| F | ID | rain | 1179 | 0.0600 | 383.89481 | 0.32561 | 0.2544 |
| F | R | rain | 1179 | 0.0075 | 383.89481 | 0.32561 | 0.2544 |
| F | INT-vk | rain | 1179 | 0.0367 | 383.89481 | 0.32561 | 0.2544 |
| F | INT | sunset | 1179 | 0.0437 | 298.23413 | 0.25296 | 0.1977 |
| F | ICS | sunset | 1179 | 0.0647 | 298.23413 | 0.25296 | 0.1977 |
| F | W | sunset | 1179 | 0.0382 | 298.23413 | 0.25296 | 0.1977 |
| F | ID | sunset | 1179 | 0.0514 | 298.23413 | 0.25296 | 0.1977 |
| F | R | sunset | 1179 | 0.0075 | 298.23413 | 0.25296 | 0.1977 |
| F | INT-vk | sunset | 1179 | 0.0442 | 298.23413 | 0.25296 | 0.1977 |
| F | INT | mean | 5895 | 0.0564 | 1770.17891 | 0.30028 | 0.2346 |
| F | ICS | mean | 5895 | 0.0498 | 1770.17891 | 0.30028 | 0.2346 |
| F | W | mean | 5895 | 0.0377 | 1770.17891 | 0.30028 | 0.2346 |
| F | ID | mean | 5895 | 0.0503 | 1770.17891 | 0.30028 | 0.2346 |
| F | R | mean | 5895 | 0.0076 | 1770.17891 | 0.30028 | 0.2346 |
| F | INT-vk | mean | 5895 | 0.0465 | 1770.17891 | 0.30028 | 0.2346 |
| HF | INT | fog | 1179 | 0.0940 | 592.10263 | 0.50221 | 0.4083 |
| HF | ICS | fog | 1179 | 0.0500 | 592.10263 | 0.50221 | 0.4083 |
| HF | W | fog | 1179 | 0.0299 | 592.10263 | 0.50221 | 0.4083 |
| HF | ID | fog | 1179 | 0.0359 | 592.10263 | 0.50221 | 0.4083 |
| HF | R | fog | 1179 | 0.0082 | 592.10263 | 0.50221 | 0.4083 |
| HF | INT-vk | fog | 1179 | 0.0840 | 592.10263 | 0.50221 | 0.4083 |
| HF | INT | morning | 1179 | 0.0528 | 230.44180 | 0.19546 | 0.1589 |
| HF | ICS | morning | 1179 | 0.0488 | 230.44180 | 0.19546 | 0.1589 |
| HF | W | morning | 1179 | 0.0379 | 230.44180 | 0.19546 | 0.1589 |
| HF | ID | morning | 1179 | 0.0493 | 230.44180 | 0.19546 | 0.1589 |
| HF | R | morning | 1179 | 0.0080 | 230.44180 | 0.19546 | 0.1589 |
| HF | INT-vk | morning | 1179 | 0.0396 | 230.44180 | 0.19546 | 0.1589 |
| HF | INT | overcast | 1179 | 0.0395 | 250.61846 | 0.21257 | 0.1728 |
| HF | ICS | overcast | 1179 | 0.0372 | 250.61846 | 0.21257 | 0.1728 |
| HF | W | overcast | 1179 | 0.0334 | 250.61846 | 0.21257 | 0.1728 |
| HF | ID | overcast | 1179 | 0.0391 | 250.61846 | 0.21257 | 0.1728 |
| HF | R | overcast | 1179 | 0.0077 | 250.61846 | 0.21257 | 0.1728 |
| HF | INT-vk | overcast | 1179 | 0.0295 | 250.61846 | 0.21257 | 0.1728 |
| HF | INT | rain | 1179 | 0.0487 | 392.01121 | 0.33249 | 0.2703 |
| HF | ICS | rain | 1179 | 0.0455 | 392.01121 | 0.33249 | 0.2703 |
| HF | W | rain | 1179 | 0.0415 | 392.01121 | 0.33249 | 0.2703 |
| HF | ID | rain | 1179 | 0.0507 | 392.01121 | 0.33249 | 0.2703 |
| HF | R | rain | 1179 | 0.0076 | 392.01121 | 0.33249 | 0.2703 |
| HF | INT-vk | rain | 1179 | 0.0329 | 392.01121 | 0.33249 | 0.2703 |
| HF | INT | sunset | 1179 | 0.0433 | 299.07634 | 0.25367 | 0.2062 |
| HF | ICS | sunset | 1179 | 0.0605 | 299.07634 | 0.25367 | 0.2062 |
| HF | W | sunset | 1179 | 0.0392 | 299.07634 | 0.25367 | 0.2062 |
| HF | ID | sunset | 1179 | 0.0575 | 299.07634 | 0.25367 | 0.2062 |
| HF | R | sunset | 1179 | 0.0075 | 299.07634 | 0.25367 | 0.2062 |
| HF | INT-vk | sunset | 1179 | 0.0409 | 299.07634 | 0.25367 | 0.2062 |
| HF | INT | mean | 5895 | 0.0557 | 1764.25044 | 0.29928 | 0.2433 |
| HF | ICS | mean | 5895 | 0.0484 | 1764.25044 | 0.29928 | 0.2433 |
| HF | W | mean | 5895 | 0.0364 | 1764.25044 | 0.29928 | 0.2433 |
| HF | ID | mean | 5895 | 0.0465 | 1764.25044 | 0.29928 | 0.2433 |
| HF | R | mean | 5895 | 0.0078 | 1764.25044 | 0.29928 | 0.2433 |
| HF | INT-vk | mean | 5895 | 0.0454 | 1764.25044 | 0.29928 | 0.2433 |
| VK_self | INT | fog | 1179 | 0.1702 | 562.75028 | 0.47731 | 0.2881 |
| VK_self | ICS | fog | 1179 | 0.1193 | 562.75028 | 0.47731 | 0.2881 |
| VK_self | W | fog | 1179 | 0.1351 | 562.75028 | 0.47731 | 0.2881 |
| VK_self | ID | fog | 1179 | 0.0826 | 562.75028 | 0.47731 | 0.2881 |
| VK_self | R | fog | 1179 | 0.0078 | 562.75028 | 0.47731 | 0.2881 |
| VK_self | INT-vk | fog | 1179 | 0.1508 | 562.75028 | 0.47731 | 0.2881 |
| VK_self | INT | morning | 1179 | 0.0584 | 123.26581 | 0.10455 | 0.0631 |
| VK_self | ICS | morning | 1179 | 0.0459 | 123.26581 | 0.10455 | 0.0631 |
| VK_self | W | morning | 1179 | 0.0472 | 123.26581 | 0.10455 | 0.0631 |
| VK_self | ID | morning | 1179 | 0.0425 | 123.26581 | 0.10455 | 0.0631 |
| VK_self | R | morning | 1179 | 0.0072 | 123.26581 | 0.10455 | 0.0631 |
| VK_self | INT-vk | morning | 1179 | 0.0518 | 123.26581 | 0.10455 | 0.0631 |
| VK_self | INT | overcast | 1179 | 0.0636 | 139.96754 | 0.11872 | 0.0717 |
| VK_self | ICS | overcast | 1179 | 0.0576 | 139.96754 | 0.11872 | 0.0717 |
| VK_self | W | overcast | 1179 | 0.0504 | 139.96754 | 0.11872 | 0.0717 |
| VK_self | ID | overcast | 1179 | 0.0484 | 139.96754 | 0.11872 | 0.0717 |
| VK_self | R | overcast | 1179 | 0.0071 | 139.96754 | 0.11872 | 0.0717 |
| VK_self | INT-vk | overcast | 1179 | 0.0661 | 139.96754 | 0.11872 | 0.0717 |
| VK_self | INT | rain | 1179 | 0.0628 | 206.19637 | 0.17489 | 0.1056 |
| VK_self | ICS | rain | 1179 | 0.0462 | 206.19637 | 0.17489 | 0.1056 |
| VK_self | W | rain | 1179 | 0.0541 | 206.19637 | 0.17489 | 0.1056 |
| VK_self | ID | rain | 1179 | 0.0497 | 206.19637 | 0.17489 | 0.1056 |
| VK_self | R | rain | 1179 | 0.0071 | 206.19637 | 0.17489 | 0.1056 |
| VK_self | INT-vk | rain | 1179 | 0.0602 | 206.19637 | 0.17489 | 0.1056 |
| VK_self | INT | sunset | 1179 | 0.0740 | 148.29648 | 0.12578 | 0.0759 |
| VK_self | ICS | sunset | 1179 | 0.0534 | 148.29648 | 0.12578 | 0.0759 |
| VK_self | W | sunset | 1179 | 0.0490 | 148.29648 | 0.12578 | 0.0759 |
| VK_self | ID | sunset | 1179 | 0.0472 | 148.29648 | 0.12578 | 0.0759 |
| VK_self | R | sunset | 1179 | 0.0072 | 148.29648 | 0.12578 | 0.0759 |
| VK_self | INT-vk | sunset | 1179 | 0.0575 | 148.29648 | 0.12578 | 0.0759 |
| VK_self | INT | mean | 5895 | 0.0858 | 1180.47648 | 0.20025 | 0.1209 |
| VK_self | ICS | mean | 5895 | 0.0645 | 1180.47648 | 0.20025 | 0.1209 |
| VK_self | W | mean | 5895 | 0.0672 | 1180.47648 | 0.20025 | 0.1209 |
| VK_self | ID | mean | 5895 | 0.0541 | 1180.47648 | 0.20025 | 0.1209 |
| VK_self | R | mean | 5895 | 0.0073 | 1180.47648 | 0.20025 | 0.1209 |
| VK_self | INT-vk | mean | 5895 | 0.0773 | 1180.47648 | 0.20025 | 0.1209 |

Conclusion ① (F): **PASS**; mean INT CCR 0.0564, random 0.0076 (7.45×), stored ICS 0.0498.

Conclusion ① (HF): **PASS**; mean INT CCR 0.0557, random 0.0078 (7.13×), stored ICS 0.0484.

Conclusion ① (VK_self): **PASS**; mean INT CCR 0.0858, random 0.0073 (11.80×), stored ICS 0.0645.

Variant check (F): both inequalities hold for 3/5 variants; failed variants: morning, sunset. The displayed primary gate uses the predeclared equal mean over five variants and does not imply uniform dominance.

Variant check (HF): both inequalities hold for 4/5 variants; failed variants: sunset. The displayed primary gate uses the predeclared equal mean over five variants and does not imply uniform dominance.

Variant check (VK_self): both inequalities hold for 5/5 variants; failed variants: none. The displayed primary gate uses the predeclared equal mean over five variants and does not imply uniform dominance.

## Table 4 — Cross-weather and clone retrieval (⑤)

PENDING: use final Night epoch-2 checkpoints C0–C4 and X3-1 after C training completes. Current C-series implementation uses APR off for C0–C4; X3-1 is the APR-on, no-ICS reference. The trained C2/C3 bases were fixed using HF stage3 FAP residuals, as documented by their own manifests; they are distinct from the stage4 fap+photo+student INT diagnostic defined for this evidence package.

## Figures and reproducibility

For each model, `part1/<model>/figures/` contains PDF and PNG for B (40 fixed random test frames), C (10 globally unique tracks at each INT e1 extreme), D (signed e1 dose response), D_absolute (magnitude supplement), and INT_quadrant (fixed 2×2 scatter). Dose error bars are RoI standard deviations, not confidence intervals; same-frame other-car references use the same dose-frame pool. Curve signs use a fixed largest-coordinate-positive basis orientation. Crop, frame, scatter and dose source records are retained. A and E are added with the GPU parts.

Raw feature NPZs, copied image data and weights remain on the server; all metric CSVs, small fitted bases, figures, source scripts, input hashes, effective arguments and logs are published. No samples, seeds, checkpoints or domains are selected using the measured outcomes. BDD unpaired domain differences include scene and identity differences; a successful probe alone would not establish weather causality.
