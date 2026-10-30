# ICS v2: APR-off single-seed Night experiments

**Training seed: 1790355234 only.** C0–C4 all disable APR and do not load APR history. Each starts from the identical HF stage3 learned weights, trains Night for two epochs, four V100s, 16 samples/GPU, two workers/GPU. APR-off refers to this Night continuation; the common HF stage3 prefix is retained. All five set causal_min_samples=32 and roi_proto_couple_ics=False.

Status: complete.

C0: no ICS. C1: old global_pc_ema, one PC, alpha=.25, loss weight=.1, bank reset to constructor state (student and teacher); all nonbank state is retained. C2/C3: frozen A-stage HF stage3 fit-video GT-box FAP nuisance-2, alpha=.25/.5, weight=.1. C4: frozen seed0 random orthogonal nuisance control, alpha=.25, weight=.1. C2–C4 store rows [2,256], initialized=True, and make zero bank update calls.

X3-1/X3-0 are reused APR-on references, not controls for the revised criterion. X3-1 disables ICS; X3-0 uses the final original ICS settings (alpha=.1, weight=.025, min_samples96). The C-series inference retains the X3 frozen student, Conv-BN fusion, thresholds .05/.30/.20, full 36-video/7286-frame Night and 33-video/6476-frame Dawn/Dusk pools, all-frame HOTA.

| Group | APR | Domain | HOTA | DetA | IDF1 | AssA | Night IDSW |
|---|---|---|---:|---:|---:|---:|---:|
| X3-1 | on | night | 33.537845 | 28.871227 | 39.966979 | 39.608983 | 3375 |
| X3-1 | on | dawn_dusk | 42.354091 | 37.919622 | 49.874666 | 47.761855 | — |
| X3-0 | on | night | 33.282830 | 28.681273 | 39.678051 | 39.275171 | 3439 |
| X3-0 | on | dawn_dusk | 42.186656 | 37.970821 | 49.703742 | 47.382786 | — |
| C0 | off | night | 33.414468 | 28.773597 | 39.839971 | 39.418914 | 3416 |
| C0 | off | dawn_dusk | 42.195528 | 37.849307 | 49.825328 | 47.543649 | — |
| C1 | off | night | 33.388818 | 28.732189 | 39.710723 | 39.447240 | 3382 |
| C1 | off | dawn_dusk | 42.239623 | 37.857058 | 49.805829 | 47.582837 | — |
| C2 | off | night | 33.666016 | 28.730627 | 40.017089 | 40.108661 | 3369 |
| C2 | off | dawn_dusk | 42.347418 | 37.855380 | 50.203219 | 47.814880 | — |
| C3 | off | night | 33.394051 | 28.584863 | 39.831658 | 39.649314 | 3387 |
| C3 | off | dawn_dusk | 42.109284 | 37.871179 | 49.613212 | 47.281407 | — |
| C4 | off | night | 33.414626 | 28.700512 | 39.826316 | 39.574424 | 3407 |
| C4 | off | dawn_dusk | 42.112312 | 37.836695 | 49.777732 | 47.372085 | — |

Predeclared revised criterion: C1 is effective if Night HOTA **or** AssA exceeds C0 by at least 0.4 points. C2/C3 must also exceed C4 in the same qualifying metric. This criterion replaces the initial APR-on/X3 comparison rule. It is descriptive for one seed, not a statistical significance claim.

- C1: criterion not met. HOTA ΔC0=-0.025650, ΔC4=-0.025808; AssA ΔC0=+0.028326, ΔC4=-0.127184.
- C2: criterion met. HOTA ΔC0=+0.251548, ΔC4=+0.251391; AssA ΔC0=+0.689747, ΔC4=+0.534237.
- C3: criterion not met. HOTA ΔC0=-0.020417, ΔC4=-0.020575; AssA ΔC0=+0.230400, ΔC4=+0.074890.

Training activity and startup verification:

- C0: 228 batches/rank, key/ref actual nonidentity ratios 0.0000/0.0000; mean weighted loss_track_trim absent (ICS off); first initialization by rank not applicable.
- C1: 228 batches/rank, key/ref actual nonidentity ratios 0.9550/1.0000; mean weighted loss_track_trim 0.024702761; first initialization by rank [1, 1, 1, 1].
- C2: 228 batches/rank, key/ref actual nonidentity ratios 0.9539/1.0000; mean weighted loss_track_trim 0.024550009; first initialization by rank fixed at startup; zero updates.
- C3: 228 batches/rank, key/ref actual nonidentity ratios 0.9529/1.0000; mean weighted loss_track_trim 0.027030631; first initialization by rank fixed at startup; zero updates.
- C4: 228 batches/rank, key/ref actual nonidentity ratios 0.9539/1.0000; mean weighted loss_track_trim 0.023402074; first initialization by rank fixed at startup; zero updates.

The actual nonidentity ratios count all training batches, independently for key and ref. A trim call returning the original values is inactive even when a trim loss is logged. The exact first50 records and any below80% alerts are in activation_first50.csv and per-rank logs.

CPU preflight verifies learned weights and all nonbank state bitwise, C1 bank-only replacement, the 32-RoI gate, finite gradients, exact operator orientation, row orthonormal error <1e-5 and principal angles <0.01°. Formal per-rank audits verify the same startup state, zero APR history and zero fixed-bank update calls. Check basis/provenance.json, basis/A1_selected_entry.json, runs/*/config_diff.json, manifests/preflight.json and manifests/final_validation.json.

The five evaluated server checkpoints, their SHA256 values and metric hashes are recorded in manifests/final_model_provenance.json; checkpoint weights stay on the server.

Historical full-run reassessment (separate legacy evaluator): [completed comparison](historical_completed/SUMMARY.md), [all stage/domain metrics](historical_completed/stage_domain_metrics.csv), [old-ICS minus no-ICS deltas](historical_completed/old_ICS_comparisons.csv). [Per-domain diagonal HOTA/AssA](historical_completed/DIAGONAL_BY_DOMAIN.md) and [C1 exact activity counts](manifests/C1_activity_summary.json) are included. Twelve missing old-domain cells were inferred; no historical training was repeated. Historical seeds, APR strengths and batch/GPU counts differ and are disclosed. The initial pre-inference table is preserved in historical/SUMMARY.md.

Only this task directory was modified. The starting snapshots are read-only symlinks to the identical original HF file. Redundant task-owned epoch1/stage00 copies and duplicate COCO exports are removed with hash records; epoch1 recovery files remain until training succeeds, while unused stage00 copies may be removed earlier for disk space. Epoch2 and the final endpoint remain on the server. Weights, NPZs and raw predictions are excluded from GitHub.
