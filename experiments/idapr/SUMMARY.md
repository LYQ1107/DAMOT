# idAPR staged experiment

Run: `/data1/liuyeqiang/idapr_20261030_153059`

**Stage A: FAIL.** STOP; publish Stage A only.

F k16 identity estimates (stage1/stage3): recovery 38.694% / 39.615%; fraction of the identity oracle 65.417% / 66.975%. FAIL: neither estimate meets both thresholds.

HF k16 identity estimates (stage1/stage3): recovery 43.223% / 44.594%; fraction of the identity oracle 77.444% / 79.900%. PASS.

Nuisance k2 recovery: F 0.288%, HF 0.685%; both pass the 10% upper limit.

The fixed protocol uses k = 2, 8, 16, car/fit videos for label-free fitting, and exact S3 all-class retrieval for the oracle regression. All recovery ratios below are the equal mean of the three per-domain ratios, not a ratio of averaged mAP values.

## Table A1 — Label-free subspace properties

Angles are median/max in degrees against the common S3 F_stage1 fit oracle. Energy shares use held-out car videos and the P1 ratio-of-sums definition; real domain shifts use Clear as the reference. Stability compares same-dimensional stage1 and stage3 estimates. Stage4 uses FAP, photo and student views; the earlier stages use FAP only.

| Model/stage | Space | k | Identity angle | Weather angle | FAP angle | Real shift energy | Identity energy | FAP energy | Stage1↔3 angle |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| F/1 | identity | 2 | 64.18/81.14 | 74.83/86.79 | 85.36/89.37 | 6.11% | 13.16% | 1.25% | 47.85/53.86 |
| F/1 | identity | 8 | 53.35/77.34 | 70.87/82.53 | 72.95/89.76 | 17.94% | 23.76% | 4.90% | 41.46/66.12 |
| F/1 | identity | 16 | 50.97/87.80 | 64.22/74.76 | 73.92/89.96 | 23.26% | 29.15% | 8.60% | 39.81/80.00 |
| F/1 | nuisance | 2 | 69.09/73.71 | 73.70/89.94 | 4.11/8.21 | 37.39% | 10.87% | 6.01% | 47.12/58.95 |
| F/1 | nuisance | 8 | 28.26/85.66 | 50.57/56.28 | 0.01/11.49 | 57.00% | 40.59% | 17.57% | 29.50/88.59 |
| F/1 | nuisance | 16 | 29.29/84.79 | 36.36/46.03 | 0.00/48.64 | 68.65% | 56.11% | 27.44% | 26.03/89.23 |
| F/3 | identity | 2 | 65.36/76.93 | 74.91/80.25 | 81.84/88.37 | 3.39% | 11.98% | 1.30% | 47.85/53.86 |
| F/3 | identity | 8 | 57.75/79.27 | 71.45/82.71 | 73.15/88.68 | 20.85% | 24.43% | 5.13% | 41.46/66.12 |
| F/3 | identity | 16 | 56.11/84.67 | 65.65/74.17 | 73.72/89.71 | 27.31% | 29.27% | 8.91% | 39.81/80.00 |
| F/3 | nuisance | 2 | 77.03/86.32 | 71.38/85.54 | 46.49/58.95 | 10.71% | 8.41% | 5.46% | 47.12/58.95 |
| F/3 | nuisance | 8 | 34.58/84.37 | 49.61/73.64 | 29.10/87.14 | 56.20% | 46.50% | 19.39% | 29.50/88.59 |
| F/3 | nuisance | 16 | 33.46/88.01 | 43.68/48.20 | 25.66/86.89 | 75.04% | 60.63% | 29.77% | 26.03/89.23 |
| F/4 | identity | 2 | 66.76/81.00 | 71.89/78.83 | 82.36/88.26 | 4.03% | 13.68% | 1.29% | — |
| F/4 | identity | 8 | 57.22/81.15 | 71.20/81.60 | 71.95/89.31 | 20.47% | 25.22% | 5.04% | — |
| F/4 | identity | 16 | 57.65/88.98 | 65.07/73.06 | 71.63/89.55 | 24.79% | 30.14% | 9.03% | — |
| F/4 | nuisance | 2 | 82.47/83.71 | 73.90/88.91 | 59.07/72.48 | 35.00% | 10.60% | 7.89% | — |
| F/4 | nuisance | 8 | 39.30/86.18 | 52.85/76.08 | 36.00/88.02 | 53.52% | 46.11% | 22.30% | — |
| F/4 | nuisance | 16 | 36.92/88.10 | 44.78/54.90 | 32.44/89.32 | 72.46% | 64.52% | 32.80% | — |
| HF/1 | identity | 2 | 65.52/82.04 | 75.48/85.85 | 83.94/87.94 | 6.44% | 12.96% | 1.25% | 36.12/39.00 |
| HF/1 | identity | 8 | 55.53/77.97 | 71.94/82.39 | 73.14/89.14 | 18.61% | 23.93% | 4.86% | 36.91/67.15 |
| HF/1 | identity | 16 | 52.25/80.84 | 64.04/76.34 | 73.51/88.89 | 24.05% | 29.14% | 8.48% | 36.17/66.81 |
| HF/1 | nuisance | 2 | 71.92/73.66 | 72.80/87.53 | 17.35/19.94 | 36.22% | 10.98% | 5.88% | 40.45/48.26 |
| HF/1 | nuisance | 8 | 32.41/84.93 | 51.48/63.00 | 16.02/77.02 | 53.43% | 39.48% | 17.85% | 27.47/81.71 |
| HF/1 | nuisance | 16 | 32.35/86.28 | 39.46/47.40 | 13.66/59.37 | 69.85% | 57.25% | 27.56% | 26.32/84.23 |
| HF/3 | identity | 2 | 70.99/89.21 | 75.62/83.13 | 81.77/88.07 | 5.70% | 13.53% | 1.29% | 36.12/39.00 |
| HF/3 | identity | 8 | 57.05/81.94 | 70.73/83.02 | 72.32/89.56 | 18.49% | 24.69% | 5.04% | 36.91/67.15 |
| HF/3 | identity | 16 | 55.98/84.34 | 62.88/74.59 | 73.16/88.37 | 25.58% | 29.71% | 8.85% | 36.17/66.81 |
| HF/3 | nuisance | 2 | 75.71/85.13 | 72.90/86.37 | 40.49/45.24 | 34.98% | 11.28% | 5.45% | 40.45/48.26 |
| HF/3 | nuisance | 8 | 33.68/67.22 | 48.79/77.54 | 29.37/80.92 | 55.23% | 47.18% | 18.54% | 27.47/81.71 |
| HF/3 | nuisance | 16 | 33.25/83.88 | 43.14/48.02 | 27.11/77.86 | 74.06% | 62.91% | 29.01% | 26.32/84.23 |
| HF/4 | identity | 2 | 61.10/71.30 | 70.85/76.38 | 82.57/89.13 | 2.55% | 13.34% | 1.23% | — |
| HF/4 | identity | 8 | 60.43/84.06 | 72.12/82.81 | 73.46/89.52 | 19.82% | 25.60% | 4.92% | — |
| HF/4 | identity | 16 | 58.78/86.09 | 65.10/75.75 | 71.04/88.93 | 25.00% | 31.08% | 9.05% | — |
| HF/4 | nuisance | 2 | 86.25/89.04 | 72.70/87.11 | 61.97/77.91 | 39.39% | 11.11% | 7.57% | — |
| HF/4 | nuisance | 8 | 39.77/78.72 | 50.93/79.79 | 34.74/89.92 | 55.55% | 45.23% | 21.74% | — |
| HF/4 | nuisance | 16 | 37.35/86.01 | 44.73/51.33 | 34.57/81.35 | 70.79% | 62.92% | 32.15% | — |

All 54 (k_id, k_nuis) combinations and complete angle lists are retained in [A1_estimates.csv](A/A1_estimates.csv); distinct subspaces are in [A1_subspaces.csv](A/A1_subspaces.csv).

## Table A2 — Restoration of old-domain forgetting

| Model | Subspace | k | Mean recovery | Fraction of identity oracle |
|---|---|---:|---:|---:|
| F | Label-free identity stage1 | 2 | 21.967% | 48.698% |
| F | Label-free identity stage1 | 8 | 25.305% | 49.453% |
| F | Label-free identity stage1 | 16 | 38.694% | 65.417% |
| F | Label-free identity stage3 | 2 | 21.988% | 48.745% |
| F | Label-free identity stage3 | 8 | 27.936% | 54.595% |
| F | Label-free identity stage3 | 16 | 39.615% | 66.975% |
| F | Label-free nuisance stage1 | 2 | 0.288% | 0.638% |
| F | Label-free nuisance stage1 | 8 | 20.687% | 40.429% |
| F | Label-free nuisance stage1 | 16 | 45.933% | 77.656% |
| F | Identity oracle | 2 | 45.108% | 100.000% |
| F | Identity oracle | 8 | 51.170% | 100.000% |
| F | Identity oracle | 16 | 59.150% | 100.000% |
| F | FAP residual PCA | 2 | -0.369% | -0.818% |
| F | FAP residual PCA | 8 | 20.117% | 39.314% |
| F | FAP residual PCA | 16 | 49.311% | 83.366% |
| F | Random mean (10) | 2 | 1.406% | 3.116% |
| F | Random mean (10) | 8 | 4.959% | 9.692% |
| F | Random mean (10) | 16 | 12.695% | 21.462% |
| HF | Label-free identity stage1 | 2 | 13.044% | 19.394% |
| HF | Label-free identity stage1 | 8 | 30.141% | 60.083% |
| HF | Label-free identity stage1 | 16 | 43.223% | 77.444% |
| HF | Label-free identity stage3 | 2 | 14.314% | 21.283% |
| HF | Label-free identity stage3 | 8 | 30.300% | 60.399% |
| HF | Label-free identity stage3 | 16 | 44.594% | 79.900% |
| HF | Label-free nuisance stage1 | 2 | 0.685% | 1.019% |
| HF | Label-free nuisance stage1 | 8 | 41.089% | 81.906% |
| HF | Label-free nuisance stage1 | 16 | 69.581% | 124.671% |
| HF | Identity oracle | 2 | 67.256% | 100.000% |
| HF | Identity oracle | 8 | 50.166% | 100.000% |
| HF | Identity oracle | 16 | 55.812% | 100.000% |
| HF | FAP residual PCA | 2 | 0.997% | 1.483% |
| HF | FAP residual PCA | 8 | 31.694% | 63.179% |
| HF | FAP residual PCA | 16 | 59.963% | 107.439% |
| HF | Random mean (10) | 2 | 1.847% | 2.747% |
| HF | Random mean (10) | 8 | 6.707% | 13.370% |
| HF | Random mean (10) | 16 | 12.621% | 22.614% |

Identity oracle self-check: **PASS**, 18/18 cells, maximum absolute discrepancy 0 (required < 1e-9). The six original mAP/forgetting fields and query counts are checked for every cell.

Per-domain values: [A2_restoration.csv](A/A2_restoration.csv). All ten random replicates: [A2_random_replicates.csv](A/A2_random_replicates.csv). Oracle regression: [oracle_self_check.csv](A/oracle_self_check.csv).

### Stage A gate

| Model | Test | Estimate stage | Recovery | Oracle fraction | Pass |
|---|---|---:|---:|---:|---|
| F | identity_k16 | 1 | 38.694% | 65.417% | FAIL |
| F | identity_k16 | 3 | 39.615% | 66.975% | FAIL |
| F | nuisance_k2 | 1 | 0.288% | — | PASS |
| HF | identity_k16 | 1 | 43.223% | 77.444% | PASS |
| HF | identity_k16 | 3 | 44.594% | 79.900% | PASS |
| HF | nuisance_k2 | 1 | 0.685% | — | PASS |

Identity gate: each model needs at least one of stage1/stage3 k16 estimates to recover ≥40% and ≥70% of the same-k identity oracle. Nuisance k2 must recover ≤10% in both models. Both estimates are retained regardless of the gate.

Selected B dimension: not selected; A failed.

## Table U — Unit and smoke checks

| Check | Status | Reason |
|---|---|---|
| U1 CPU loss/gradient tests | NOT RUN | Stage A did not pass; stopped as instructed |
| U2 default-off bitwise parity | NOT RUN | Stage A did not pass; stopped as instructed |
| U3 B1/B2/B3 smoke training | NOT RUN | Stage A did not pass; stopped as instructed |

## Table B — Night training and forgetting

| Group | Status | Reason |
|---|---|---|
| B0 extra Overcast/Rainy evaluation | NOT RUN | Stage A did not pass; stopped as instructed |
| HF stage3 extra old-domain evaluation | NOT RUN | Stage A did not pass; stopped as instructed |
| B1 | NOT RUN | Stage A did not pass; stopped as instructed |
| B2 | NOT RUN | Stage A did not pass; stopped as instructed |
| B3 | NOT RUN | Stage A did not pass; stopped as instructed |

Stage B gate: not evaluated because Stage B has not run. No new training seeds were added.

## Protocol clarifications and deviations

- HF feature files are located under `/data1/liuyeqiang/damot_ics_followup_20261029/results/feats`, as recorded by the existing probe analysis; F uses the supplied RUN directory. All 40 input NPZ hashes and resolved paths are recorded in `manifests/feature_inputs.json`.
- The label-free estimators use car fit RoIs only. The diagnostic oracle and A2 retrieval retain all S3 classes and the common F_stage1 reference so the requested oracle regression has exactly the historical protocol. Test energies use car only, as in P1.
- Image IDs are namespaced by weather/video/frame only if the stored IDs collide between physical images. Each fit records whether this was necessary; no RoIs are removed.
- Random controls use seed 0, ten draws per dimension, shared across F and HF; all replicates are reported. Stage A never loads a model on a GPU.

Source code, inputs, fixed gate thresholds and logs are included in the publication bundle. Weights and feature NPZ files remain on the server.
