# ICS mechanism experiments

Run: `/data1/liuyeqiang/ics_quick_20261029_160807`

All metric values and differences are percentage points. All mandatory X1, X2 and X3 cells are complete.

## Table 1 — X1 gradients

| Model | Domain | Loss key present | Key RoIs ≥96 | Parameter cosine, mean ± SD | Norm ratio, mean ± SD | Total vs original cosine | Key difference energy in E |
|---|---|---:|---:|---:|---:|---:|---:|
| F | night | 20/20 | 3/20 | 0.994003 ± 0.005028 | 0.994872 ± 0.041464 | 0.999996337 | 0.494345 ± 0.200268 |
| F | overcast | 20/20 | 16/20 | 0.987775 ± 0.006732 | 0.968308 ± 0.046405 | 0.999993324 | 0.423071 ± 0.135528 |
| HF | night | 20/20 | 7/20 | 0.992199 ± 0.005795 | 0.969096 ± 0.035513 | 0.999995652 | 0.474954 ± 0.156351 |
| HF | overcast | 20/20 | 16/20 | 0.983566 ± 0.011123 | 0.949941 ± 0.039079 | 0.999991134 | 0.434533 ± 0.151011 |

Interpretation under the prespecified ≥0.99 cosine rule: 甲 in F/night, HF/night. Below the 甲 threshold in F/overcast, HF/overcast. This is a local gradient test and does not establish equivalence of trained models.

Loss-key activation and a non-identity trim are different in this implementation: it still computes the additional loss when the key batch has fewer than 96 RoIs. Key trimming is then the identity, although reference trimming may remain active. All 20 batches per cell are retained; none are filtered by the measured cosine or RoI threshold.

[Per-batch gradients](X1/grad_compare.csv) and [full gradient summaries](X1/grad_summary.csv) include both parameter and key-embedding measurements, with means and standard deviations.

## Table 2 — X2 inference intervention

| Model | Domain | β | HOTA | DetA | IDF1 | AssA | ΔAssA |
|---|---|---:|---:|---:|---:|---:|---:|
| F | night | 0.0 | 34.280073 | 28.392947 | 40.773418 | 41.990505 | +0.000000 |
| F | night | 1.0 | 33.545412 | 28.371480 | 39.635446 | 40.256029 | -1.734476 |
| F | overcast | 0.0 | 37.333094 | 29.968487 | 42.198259 | 46.815848 | +0.000000 |
| F | overcast | 1.0 | 35.576689 | 29.925115 | 39.859030 | 42.582729 | -4.233119 |
| HF | night | 0.0 | 34.851921 | 29.745117 | 41.564756 | 41.468328 | +0.000000 |
| HF | night | 1.0 | 34.416454 | 29.685460 | 40.769627 | 40.532198 | -0.936131 |
| HF | overcast | 0.0 | 37.584746 | 30.265997 | 42.362813 | 46.987987 | +0.000000 |
| HF | overcast | 1.0 | 36.924725 | 30.249039 | 41.592235 | 45.385918 | -1.602069 |

Interpretation: No tested cell supports 乙 under the prespecified AssA-increase rule. Removing the two directions completely at inference does not by itself identify the effect of soft training shrinkage at α=0.1.

Principal angles between each Clear PCA basis and its checkpoint bank: F: 14.770° and 82.348° (7956 clean car embeddings); HF: 16.550° and 82.124° (7956 clean car embeddings).

Detector control: β=0 and β=1 have bitwise identical detector outputs for every Overcast frame (4448/4448 for each model); the intervention changes the association embeddings.

F uses the user-approved current-environment β=0 baseline for paired intervention differences. The historical baseline mismatch is retained below; it is not counted as an exact historical reproduction.

F historical-baseline checks (using the archived cuDNN benchmark setting):

| Domain | Metric | Archived F | Current V100 F | Difference |
|---|---|---:|---:|---:|
| overcast | HOTA | 37.291054916 | 37.333094255 | +0.042039339 |
| overcast | DetA | 29.971056214 | 29.968486500 | -0.002569714 |
| overcast | IDF1 | 42.083274985 | 42.198258759 | +0.114983774 |
| overcast | AssA | 46.708384349 | 46.815848267 | +0.107463918 |
| night | HOTA | 34.290420755 | 34.280073441 | -0.010347314 |
| night | DetA | 28.387473648 | 28.392946579 | +0.005472931 |
| night | IDF1 | 40.782635280 | 40.773417797 | -0.009217483 |
| night | AssA | 42.022990753 | 41.990504740 | -0.032486013 |

Full X2 metrics and intervention differences, including car and the other classes: [intervention.csv](X2/intervention.csv).

## Table 3 — X3 Night adaptation (2 epochs)

X3-0 uses the original ICS setting; X3-1 disables ICS; X3-2 sets α=0 while retaining the extra loss; X3-3 freezes a random orthogonal two-dimensional basis generated with seed 0. All four groups disable APR–ICS coupling, restore the same three-domain APR history and use the same training seed.

| Group | Seed | Domain | HOTA | DetA | IDF1 | AssA | Old-domain HOTA forgetting |
|---|---:|---|---:|---:|---:|---:|---:|
| X3-0 | 1790355234 | night | 33.282830 | 28.681273 | 39.678051 | 39.275171 | — |
| X3-0 | 1790355234 | dawn_dusk | 42.186656 | 37.970821 | 49.703742 | 47.382786 | 0.124296 |
| X3-1 | 1790355234 | night | 33.537845 | 28.871227 | 39.966979 | 39.608983 | — |
| X3-1 | 1790355234 | dawn_dusk | 42.354091 | 37.919622 | 49.874666 | 47.761855 | -0.043139 |
| X3-2 | 1790355234 | night | 33.263440 | 28.515266 | 39.807984 | 39.480546 | — |
| X3-2 | 1790355234 | dawn_dusk | 42.337794 | 37.909027 | 50.004104 | 47.716544 | -0.026842 |
| X3-3 | 1790355234 | night | 33.681751 | 28.656312 | 40.270597 | 40.223440 | — |
| X3-3 | 1790355234 | dawn_dusk | 42.108966 | 37.825210 | 49.748656 | 47.298246 | 0.201986 |

Pairwise HOTA differences (X3-0 minus each alternative):

| Domain | vs ICS off (X3-1) | vs α=0 (X3-2) | vs random (X3-3) |
|---|---:|---:|---:|
| night | -0.255016 | +0.019390 | -0.398921 |
| dawn_dusk | -0.167435 | -0.151138 | +0.077690 |

Interpretation under the prespecified single-seed rule: night: 甲 (X3-0 and α=0 differ by at most 0.3 HOTA); dawn_dusk: 无差异 across all four groups; compatible with 甲, without evidence of an ICS gain. Conditions are checked independently; overlapping patterns do not uniquely identify a mechanism. Differences within ±0.3 HOTA are treated as no difference.

Comparisons exceeding the 0.3 HOTA threshold: night: frozen random is +0.398921 HOTA relative to original ICS. These are single-seed observations.

Original ICS and ICS off also differ by at most 0.3 HOTA in both domains. Compatibility with 甲 therefore does not establish that adding the extra loss provides a gain over disabling ICS.

The HF stage3 Dawn/Dusk baseline (OVERALL) is HOTA 42.310952, DetA 38.126883, IDF1 50.225128 and AssA 47.462822. Forgetting is baseline minus the final value; negative values are retained and indicate improvement. Full per-class metrics and forgetting values are in [x3_summary.csv](X3/x3_summary.csv).

The optional second seed has not been run; the mandatory four groups take priority under shared GPU availability.

## Protocol details and deviations

Both OVERALL and car metrics are preserved in the CSV files; the compact tables above display OVERALL. X1 records bank initialization and updates, alpha, dimension and loss activation in per-batch CSVs. X2 uses full videos and checks β=0 against the historical frozen metrics before β=1. X3 uses identical SHA256-identified archived APR history with three domains.

- **X1:** Only embed_head parameters retain requires_grad; other parameters are held constant for autograd. Only the requested partial derivatives (embed_head parameters and key embeddings) are measured. This avoids retaining unnecessary backbone graphs for batch16 on V100; no forward values, loss function, train mode, samples or weights are changed.
- **X1 F/night:** Initial run did not call roi_proto_start_stage, causing APR candidate collection to warn on empty history. APR is not part of either measured association loss. This cell retains its prespecified 20 batches and all association gradients; subsequent cells initialize the empty APR collection buffers explicitly.
- **X1:** Report loss-key activation separately from key RoI sample threshold. The implementation emits loss_track_trim even when key RoIs <96; in this case key trim is identity but reference trim may still be active. Loss computation and effective key shrinkage are distinct events in the implementation, so both are reported.
- **X2:** Use native frozen inference via equivalent Python evaluation wrapper with metric bbox+track and all-frame HOTA correction. The historical F/HF reference numbers use all-frame HOTA; the original repository default indexes only frames containing GT and must be corrected for exact parity. Detector .05, tracker .30/.20, fusion, classes, image sizes and tracker logic are preserved. F original runtime configs are read-only because configs/mot is absent from commit ebc3190.
- **X3:** Use one archived HF stage3 rank0 APR history identically in all four groups and ranks. Original rank-local APR histories are available. This fixes a shared starting library across groups, with its SHA256 recorded; no rebuild needed.
- **X3:** Replace HF run-root audit hook with local audit hook, retain finite-loss and EMA hooks; keep find_unused_parameters=True as actual HF training. Prevents writes to the original HF directory; the generic seq3 script otherwise overrides this HF execution setting.
- **X3 storage:** After a group finishes both evaluations, remove only its generated epoch1 and stage00 source copies; retain epoch2, final endpoint and stage3 starting copy. Storage was limited when the run was prepared (about 18 GiB free). This does not change training or reported checkpoints and avoids accumulating redundant weights. No other task files are modified.
- **Input image access:** Independent read-only SSHFS session under this run, exposed through private data_views symlinks to the same original image files. Shared NAS mount briefly returned ENOTCONN during both X2 baselines. Existing completed video predictions were retained; incomplete videos are retried with the same inputs. No shared mount or image files were modified.
- **X2 F first baseline:** First F beta0 used cuDNN benchmark=False and found a small mismatch with archived metrics; F beta1 was held until the discrepancy was investigated. The archived F command explicitly uses cuDNN benchmark=True. A separate corrected full-domain run used this setting, with the same discrepancy. The first baseline and all mismatch metrics are preserved. The archival F reference only lists OVERALL and car, so parity is checked for those available categories. HF has all six categories and passes all of them.
- **X2 F historical reproducibility gate:** Corrected full-domain F beta0 still differed from the historical frozen scores; F beta1 was paused until the user approved the current-environment paired baseline. Restoring cuDNN benchmark=True left the reported metrics unchanged. Disabled intervention output is bitwise identical to the native class, yet archived detections already differ slightly before association. A cross-runtime numerical difference or historical provenance gap remains possible, not established. All results and parity deltas are retained; no values are adjusted to match the archive.
- **X2 F user-approved protocol change:** Use the current V100 F beta0 as the paired intervention baseline in both domains; keep the archived F discrepancy table. User explicitly approved this on 2026-10-29 after the corrected full-domain baseline still differed by +0.042 HOTA / +0.115 IDF1 / +0.107 AssA. Metrics are never altered to force historical equality. HF retains its exact historical-baseline gate.

## Provenance

See `inputs.txt`, `manifests/`, per-cell resolved configs, `X3/*/config_diff.json`, and `logs/`. Checkpoints and feature NPZ files stay on the server and are excluded from GitHub.
