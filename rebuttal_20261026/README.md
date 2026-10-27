# DAMOT rebuttal experiments (V100, 2026-10-26/27)

This directory is the curated, verifiable delivery of the four rebuttal
experiments run on a Tesla V100-DGXS-32GB. The parent repository already
contains the DAMOT model implementation; this bundle adds the exact runner
scripts, resolved configuration snapshots, compact provenance, and the
result tables. Machine-local paths in copied CSV/JSON/config snapshots are
redacted to `RUN_ROOT`, `NAS_ROOT`, `NAS_AVIS_ROOT`, or `USER_HOME`; metric
values, hashes, frame counts, and checkpoint identifiers are unchanged.

The original run directory was `/data1/liuyeqiang/damot_rebuttal_20261026`.
NAS inputs and historical checkpoints were read-only. No new source-domain
training, full target adaptation, multi-seed accuracy study, parameter
search, or plots were run.

## What was measured

### Experiment 1 — old-domain association and forgetting

The final Night epoch-5 states of `N_old`, `F_old`, and `F` were compared on
Overcast, Rainy, and Dawn-Dusk. Existing qualified predictions and complete
precision CSVs were reused where valid; the official evaluator was rerun to
repair the all-frame time index. The three configurations are not a strict
APR-only comparison: batch size, nominal GPU count, and (for `F`) the ICS
estimator differ. No identity-feature extraction was performed.

| model | FinalOld IDF1 (%) | FinalOld AssA (%) | AvgF-IDF1 (pp) | AvgF-AssA (pp) | old-domain IDSw | IDSw/1000GT |
|---|---:|---:|---:|---:|---:|---:|
| N_old | 45.552278886 | 47.836913791 | 0.455794099 | 1.899441942 | 5562 | 34.768367162 |
| F_old | 45.394536561 | 48.295537499 | 0.393332295 | 0.975906989 | 5895 | 36.849968432 |
| F | 45.516186077 | 48.388069083 | 0.232338218 | 0.938922275 | 5340 | 33.380632982 |

See `results/exp1/` for the full OVERALL/car/domain/stage tables and the
configuration caveats.

### Experiment 2 — Clear holdout retention

The fixed `clear_holdout_45v` set contains 45 videos and 9,103 frames. It is
the specified 45-video annotation file (SHA256
`5afd6fafb2f9c6f3c8a48852b6841eefafabfc4e8877a5c4ffa52d1b6a6e31bd`) and
was not used for adaptation, threshold selection, or checkpoint selection.
All five frozen student evaluations use native QDTrack, batch 1, FP32,
Conv-BN fusion, detector score threshold 0.05, tracker init threshold 0.30,
and object threshold 0.20.

| model | HOTA (%) | IDF1 (%) | AssA (%) | ΔHOTA (pp) | ΔIDF1 (pp) | ΔAssA (pp) |
|---|---:|---:|---:|---:|---:|---:|
| S (epoch 20) | 37.890270267 | 43.494500604 | 45.218384463 | 0 | 0 | 0 |
| F after Overcast (epoch 5) | 39.764871093 | 46.570151424 | 46.722640651 | 1.874600826 | 3.075650820 | 1.504256188 |
| F after Rainy (epoch 5) | 41.266379521 | 48.362678329 | 48.212331825 | 3.376109254 | 4.868177725 | 2.993947363 |
| F after Dawn-Dusk (epoch 5) | 41.050909264 | 48.044557452 | 47.472824886 | 3.160638997 | 4.550056849 | 2.254440423 |
| F after Night (epoch 5) | 41.109585963 | 48.483072623 | 47.036833985 | 3.219315696 | 4.988572019 | 1.818449522 |

The complete aggregate and per-video tables are in `results/exp2/`.

### Experiment 3 — Night teacher pseudo-label quality

The same 7,286 Night frames and 36 videos were evaluated at e0/e1/e3/e5
(29,144 unique checkpoint-frame results). e0 is the teacher after Dawn-Dusk,
not the original Clear source. The real
`teacher_detector.extract_feat -> RPN -> RoI -> confidence filter` branch was
used with the archived strict `score > 0.7` filter; GT was used only for
post-hoc one-to-one IoU matching.

| timepoint | TP | FP | FN | GT | pseudo | precision | recall | pseudo/frame |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| e0 | 14,293 | 642 | 66,120 | 80,413 | 14,935 | 0.957013727 | 0.177744891 | 2.049821575 |
| e1 | 14,894 | 675 | 65,519 | 80,413 | 15,569 | 0.956644614 | 0.185218808 | 2.136837771 |
| e3 | 16,215 | 738 | 64,198 | 80,413 | 16,953 | 0.956467882 | 0.201646500 | 2.326791106 |
| e5 | 17,483 | 824 | 62,930 | 80,413 | 18,307 | 0.954989895 | 0.217415095 | 2.512626956 |

Precision decreases monotonically while recall increases; FP also increases.
This trajectory alone does not rule out all error-propagation mechanisms.
Detailed class/video tables and coordinate/filtering notes are in
`results/exp3/`.

### Experiment 4 — V100 efficiency and resource cost

Adaptation timing uses FP32, a fixed batch of four key-reference pairs, 20
warm-up steps, 100 measured steps, three repeats, real forward/loss/backward/
optimizer/EMA, and synchronized window boundaries. `pair/s = step/s × 4`.
The original archived table is retained in `results/exp4/efficiency_summary.csv`.

| group | step/s | pair/s | peak allocated (MiB) | peak reserved (MiB) |
|---|---:|---:|---:|---:|
| baseline | 0.563639877 | 2.254559507 | 8961.8623 | 11416 |
| +FAP | 0.218886216 | 0.875544864 | 8961.2236 | 11412 |
| +FAP+ICS | 0.263066090 | 1.052264361 | 8961.3496 | 11412 |
| +FAP+ICS+APR | 0.261665176 | 1.046660705 | 8961.4438 | 11412 |

The apparent original `+FAP+ICS` speedup was not reproduced in a strict
same-process interleaved rerun on GPU 1. The retest alternated
`FAP -> FAP+ICS` three times under the same 512-frame stream:

| group | mean pair/s | sample std | peak allocated (MiB) | peak reserved (MiB) |
|---|---:|---:|---:|---:|
| +FAP | 0.989247093 | 0.019559706 | 8961.3066 | 11412 |
| +FAP+ICS | 0.974920402 | 0.025712969 | 8961.3612 | 11412 |

The interleaved delta is -0.014326691 pair/s (ICS about 1.45% slower), so
the earlier +20% difference is treated as process/order variance rather than
an ICS benefit. Night batch-4 did not reach `min_samples=96`, so covariance
and eigendecomposition calls were zero in the main throughput window; the
active dense-frame module profile is reported separately in
`module_time_breakdown.csv`.

Deployment inference uses the final F student, batch 1, FP32, Conv-BN fusion,
eight fixed complete Night videos (1,621 frames), 20 warm-up frames, and
three repeats:

* **simple_test + association:** `1621 / 74.123932756 = 21.869115724 FPS`.
  Input preprocessing and H2D are outside this timer; the precise label is
  `simple_test+association (H2D excluded)`.
* **end-to-end:** `1621 / 136.504614793 = 11.876173863 FPS`, including JPEG
  read, resize/normalize/pad, H2D, model, and association.
* Peak deployment memory: 534.8882 MiB allocated, 584 MiB reserved.

FAP/ICS/APR are adaptation-only branches and are explicitly **not executed**
in the deployed student forward. Therefore no fabricated per-module
deployment FPS is reported.

## How the run was executed

The main runner scripts are under `scripts/`:

* `collect_exp1.py` — reuse provenance, stage/domain aggregation, IDSw/GT and
  forgetting statistics;
* `evaluate_clear.py` — frozen Clear holdout student evaluation;
* `diagnose_teacher.py` and `check_teacher_parity.py` — teacher branch,
  filtering, coordinate parity, and post-hoc matching;
* `profile_damot.py` — adaptation throughput, module profiling, prototype
  storage, and deployment inference;
* `retest_fap_ics_interleaved.py` — strict same-process FAP/ICS retest;
* `aggregate_results.py` and `validate_delivery.py` — aggregation and final
  acceptance checks.

The original command sequence is preserved in `scripts/reproduce.sh`; it
expects the local cached images/checkpoints and the isolated environment from
the run. `scripts/README.md` documents the individual commands and the
required data/checkpoint inputs. Resolved snapshots used for loading are in
`configs/`; model weights, image caches, and annotations are intentionally not
part of this Git repository.

## Excluded files

The GitHub delivery deliberately excludes checkpoint files, image/annotation
caches, raw prediction pickles, reconstructed prototype pickles, Python byte
code, and multi-gigabyte profiler histories. Their existence and hashes are
recorded in the local run manifests, while compact result tables and timing
windows remain here for review.
