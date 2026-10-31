# HF + online intervention-residual ICS

Status: **complete**. Single training seed **1790355234**, identical to HF.

Observed paired result: final OVERALL four-domain mean HOTA 38.387680 vs HF 38.699879 (delta -0.312199). Night HOTA changes by +0.174578; the three-old-domain HOTA mean changes by -0.474458. HOTA AvgF is 0.578167 vs HF -0.015193 (delta +0.593360; larger means more forgetting). The final four-domain average does not improve on this single paired seed.

Baseline: /data1/liuyeqiang/damot_apr_complement_hf_20261028_1420. Clear source S → Overcast → Rainy → Dawn/Dusk → Night, five epochs/domain, four V100s, FP32, 16 samples/GPU, two workers/GPU. Geometry, student low-frequency transform, contrastive high-frequency noise, APR and beta=1 coupling are retained. Only the requested ICS settings change; config_diff.json lists every output-path and model difference.

Online update: normalized student key-positive embedding minus normalized clean student embedding at exactly the same boxes; detached residuals, no mean subtraction or residual renormalization, covariance EMA .98, shrinkage0, two PCs, HF PC EMA .9 followed by QR, alpha .25, loss weight .1, min_samples32. The APR basis reader and trim operator are unchanged.

Preflight:

User-authorized gate correction: the 45° condition compares bank PC1 with the full offline two-dimensional FAP subspace. Its principal angle is33.756089°, so stage1 passes together with the HOTA delta−0.307201. The original PC1/PC1 diagnostic82.480923° caused an earlier stop and remains recorded; the user explicitly chose the subspace definition and authorized continuation after seeing both values. Original stop/protocol/validation evidence is archived at episode/registered_axis_stop_20261031T074135Z. The same single-seed run continues from the complete stage1 boundary; no stage1 retraining or basis/strength retuning. Details: manifests/angle_clarification.json and manifests/stage_boundary_recovery.json.

- U1: passed; total and all component losses bitwise equal, 0.7292587757110596.
- U2: principal angles 65.609289°, 74.419794°; reported without an angle threshold.
- U3: passed; paired disposable HF/new50-iteration runs, iteration-time increase 0.8468%.
  New key/ref actual nonidentity rates 99.5000%/100.0000%; mean weighted loss_track_trim 0.077684235; first initialization by rank [1, 1, 1, 1].
  Mean critical-rank iteration time on iterations11–50: HF 22.212150s, INTICS 22.400244s.
  Training-step timing excluding batch fetch and the independent CSV write: HF 22.112865s, INTICS 22.050363s, signed change -0.2826%. HF logs were local and candidate logs were on NAS; these are single paired timing measurements.

The original HF metadata uses cumulative teacher scale_factor and local student scale_factor=1; geometry assertions compare shapes, flips, exact box coordinates and the equivalent composed scale, while retaining the original metadata. Both views remain on identical pixels/geometry. Fixed-batch cache is disposable and stored in this task’s /dev/shm cache symlink.

Stage1 gate (user-clarified angle definition): OVERALL HOTA delta <=−0.5, or the principal angle between bank PC1 and the offline FAP top2 subspace >45°. Offline directions use only fit videos, normalized paired FAP/clean residuals without centering. The initial gate inference ran while the four training ranks retained live model/RNG state; the earlier axis-based stop and subsequent authorized stage-boundary recovery are disclosed below.

Stage1 decision: **PASS**; both conditions passed under the user-clarified angle definition.
Overcast OVERALL HOTA 36.474774 vs HF 36.781975 (delta -0.307201). Bank PC1/offline PC1 angle 82.480923°; PC1/offline2D angle 33.756089°; two subspace principal angles [33.453465219188544, 88.357491312671].
Raw decision: episode/stage1_gate.json.

Stage1 P1 energies use test videos; offline FAP directions use fit videos only.

| Bank directions | Cross-domain shift energy | Identity energy |
|---|---:|---:|
| Both PCs | 8.5078% | 9.1129% |
| PC1 | 7.1322% | 5.1749% |
| PC2 | 1.3756% | 3.9379% |

Stage4 P1 energies use test videos; offline FAP directions use fit videos only.

| Bank directions | Cross-domain shift energy | Identity energy |
|---|---:|---:|
| Both PCs | 12.7867% | 8.9356% |
| PC1 | 6.5625% | 3.7846% |
| PC2 | 6.2242% | 5.1510% |
Stage4 bank PC1/offline PC1 angle 76.812034°; PC1/offline2D angle 55.682468°; two subspace principal angles [35.625802516666745, 86.4290772065204].

| Stage | Domain | Category | HOTA | DetA | IDF1 | AssA | ΔHOTA vs HF |
|---:|---|---|---:|---:|---:|---:|---:|
| 1 | overcast | car | 40.115006 | 31.828057 | 46.316529 | 50.727402 | -0.331385 |
| 1 | overcast | OVERALL | 36.474774 | 27.820921 | 40.979041 | 48.055223 | -0.307201 |
| 2 | overcast | car | 41.624177 | 34.264639 | 48.476561 | 50.793215 | +0.122118 |
| 2 | overcast | OVERALL | 37.745290 | 29.811947 | 42.692232 | 48.092835 | +0.083264 |
| 2 | rainy | car | 44.654869 | 37.388825 | 52.209789 | 53.826655 | +0.220474 |
| 2 | rainy | OVERALL | 39.687463 | 31.682585 | 45.029037 | 50.303625 | +0.160443 |
| 3 | overcast | car | 41.871389 | 34.658919 | 48.829748 | 50.842152 | +0.424831 |
| 3 | overcast | OVERALL | 37.744872 | 29.824016 | 42.645353 | 48.099791 | +0.217188 |
| 3 | rainy | car | 45.776325 | 38.561420 | 53.664463 | 54.983519 | +0.693125 |
| 3 | rainy | OVERALL | 40.453673 | 32.237209 | 46.101461 | 51.495287 | +0.524635 |
| 3 | dawn_dusk | car | 45.974100 | 43.384095 | 55.145005 | 49.107677 | -0.037906 |
| 3 | dawn_dusk | OVERALL | 42.059757 | 38.072792 | 49.543029 | 46.930938 | -0.251194 |
| 4 | overcast | car | 41.323554 | 34.914719 | 48.249371 | 49.153023 | -0.096882 |
| 4 | overcast | OVERALL | 37.358587 | 30.100515 | 42.230307 | 46.681756 | -0.226159 |
| 4 | rainy | car | 43.897222 | 37.414017 | 50.588600 | 51.961824 | -1.022127 |
| 4 | rainy | OVERALL | 38.866968 | 31.248980 | 43.490642 | 48.883987 | -0.876565 |
| 4 | dawn_dusk | car | 46.058383 | 42.828268 | 55.585137 | 50.053778 | -0.229428 |
| 4 | dawn_dusk | OVERALL | 42.298665 | 37.557340 | 50.199171 | 48.223685 | -0.320651 |
| 4 | night | car | 37.609735 | 33.224227 | 45.376338 | 43.203836 | +0.201644 |
| 4 | night | OVERALL | 35.026499 | 29.826214 | 41.643568 | 41.783285 | +0.174578 |

Final scores use R4j; diagonals use Rjj. AIP averages the four seen-domain stage averages. AvgF = mean over old domains of max(Rij, i=j..3)−R4j, without clipping. FinalOld averages R4j for the three old domains. BWT averages R4j−Rjj. Full precision and paired deltas: results/continual_metrics.csv and results/paired_continual_deltas.csv.

| Model | Category | Metric | Final mean | Diagonal mean | AIP | AvgF | FinalOld | BWT |
|---|---|---|---:|---:|---:|---:|---:|---:|
| HF | OVERALL | HOTA | 38.699879 | 38.367967 | 38.499734 | -0.015193 | 39.982532 | 0.442549 |
| HF | OVERALL | DetA | 32.414216 | 32.098110 | 31.267772 | -0.013488 | 33.303916 | 0.421475 |
| HF | OVERALL | IDF1 | 44.586662 | 44.447568 | 43.890315 | 0.450617 | 45.593965 | 0.185460 |
| HF | OVERALL | AssA | 47.070279 | 46.510848 | 48.022717 | -0.051094 | 48.937596 | 0.745908 |
| HF | car | HOTA | 42.508922 | 42.075221 | 42.526032 | -0.010110 | 44.209199 | 0.578268 |
| HF | car | DetA | 37.302104 | 36.707162 | 36.135765 | -0.154854 | 38.679074 | 0.793256 |
| HF | car | IDF1 | 50.067555 | 49.771552 | 49.714771 | 0.510150 | 51.658918 | 0.394670 |
| HF | car | AssA | 49.256761 | 48.813726 | 50.602286 | 0.168903 | 51.406645 | 0.590713 |
| INTICS | OVERALL | HOTA | 38.387680 | 38.312123 | 38.416233 | 0.578167 | 39.508073 | 0.100742 |
| INTICS | OVERALL | DetA | 32.183262 | 31.850628 | 31.032364 | 0.409061 | 32.968945 | 0.443512 |
| INTICS | OVERALL | IDF1 | 44.390922 | 44.298669 | 43.831803 | 0.805534 | 45.306707 | 0.123005 |
| INTICS | OVERALL | AssA | 46.393178 | 46.768268 | 48.122159 | 0.912196 | 47.929809 | -0.500119 |
| INTICS | car | HOTA | 42.222224 | 42.088428 | 42.504339 | 0.780885 | 43.759720 | 0.178395 |
| INTICS | car | DetA | 37.095308 | 36.456301 | 35.904560 | 0.482476 | 38.385668 | 0.852009 |
| INTICS | car | IDF1 | 49.949861 | 49.761915 | 49.788993 | 1.072036 | 51.474369 | 0.250595 |
| INTICS | car | AssA | 48.593115 | 49.216393 | 50.818725 | 1.254908 | 50.389542 | -0.831036 |

| Category | Metric | ΔFinal mean | ΔDiagonal mean | ΔAIP | ΔAvgF | ΔFinalOld | ΔBWT |
|---|---|---:|---:|---:|---:|---:|---:|
| OVERALL | HOTA | -0.312199 | -0.055844 | -0.083501 | +0.593360 | -0.474458 | -0.341808 |
| OVERALL | DetA | -0.230954 | -0.247482 | -0.235408 | +0.422549 | -0.334971 | +0.022037 |
| OVERALL | IDF1 | -0.195740 | -0.148899 | -0.058513 | +0.354917 | -0.287258 | -0.062455 |
| OVERALL | AssA | -0.677101 | +0.257420 | +0.099442 | +0.963290 | -1.007787 | -1.246027 |
| car | HOTA | -0.286698 | +0.013207 | -0.021693 | +0.790995 | -0.449479 | -0.399873 |
| car | DetA | -0.206797 | -0.250861 | -0.231205 | +0.637330 | -0.293406 | +0.058753 |
| car | IDF1 | -0.117693 | -0.009636 | +0.074222 | +0.561886 | -0.184549 | -0.144076 |
| car | AssA | -0.663646 | +0.402666 | +0.216439 | +1.086005 | -1.017103 | -1.421749 |
Probe results: [stage1 P1](probe/stage1/P1.json), and (if stage4 is reached) [stage4 P1](probe/stage4/P1.json), [Table1 retrieval](probe/stage4/evidence/part1/INTICS/cross_view_retrieval.csv), [Table2b deletion](probe/stage4/evidence/part1/INTICS/deletion.csv) and [FigureC crops](probe/stage4/evidence/part1/INTICS/figures/C_INT_crops.pdf). Bank diagnostics read the rank0 student bank from the frozen checkpoint; all four rank-local runtime states are retained as in HF. The actual bank_pcs.npy and offline_fap_k2.npy with SHA256 hashes accompany each P1 report. Random-basis probe controls use the historical fixed seed0; they are not additional training seeds.

Evidence definitions: P1 angles compare the bank with the fit-video FAP-only offline basis. The evidence package INT basis combines FAP/photo/student residuals; its ICS basis is the actual frozen bank. FigureC shows extrema of the offline three-view INT first axis, with ten distinct tracks at each end.

| Table1 view | Cross-view identity mAP | Ratio to clean |
|---|---:|---:|
| clean | 57.759908 | 1.000000 |
| fap | 57.035253 | 0.987454 |
| photo | 57.490424 | 0.995334 |
| student | 56.876646 | 0.984708 |

| Table2b deleted basis | Weather balanced accuracy | Night/Clear balanced accuracy | Identity mAP | ΔIdentity mAP | Fisher drop |
|---|---:|---:|---:|---:|---:|
| Full | 36.617803 | 83.552599 | 57.759908 | +0.000000 | 0.000000% |
| INT | 34.977799 | 79.779468 | 57.760758 | +0.000850 | 34.852090% |
| ICS | 36.417354 | 82.859492 | 57.182154 | -0.577754 | 14.848575% |
| W | 23.782671 | 55.917213 | 57.933952 | +0.174044 | 61.780591% |
| ID | 36.283715 | 83.420024 | 58.216653 | +0.456745 | -24.476819% |
| R | 36.641959 | 83.506571 | 57.760083 | +0.000175 | 0.176775% |

Bank interpretation (inference from these probes): the final online bank has partial intervention alignment and retains identity information. Deleting its two directions changes identity mAP by -0.577754 points and reduces weather Fisher by 14.848575%; deleting the offline three-view INT directions changes identity mAP by +0.000850 points and reduces Fisher by 34.852090%. Historical evidence checks: cross-view retrieval passes, INT deletion passes, ICC fails (INT−ID -0.007435, required >=0.3). These probe checks describe the measured representation; the requested training stop was evaluated at stage1.

Evaluation scheduling: immutable completed-stage snapshots were evaluated on additional wholly idle GPUs while the same four-GPU training continued, with at most two inference jobs concurrently. The final controller reuses complete cells and verifies their checkpoint hashes. Records: [{"stage": 2, "status": "complete"}, {"stage": 3, "status": "complete"}]. Training settings and the mandatory stage1 gate were retained.

Runtime deviation: user-authorized continuation from the complete stage1 checkpoint and all four final rank states after correcting the gate interpretation. Restoration audits verify model parameters, rank-local buffers, APR runtime state, counters and Python/NumPy/Torch/CUDA RNGs before any stage2 update. Verified ranks at report time: [0, 1, 2, 3]. New-domain workers and optimizer are created by the unchanged HF stage runner; FAP audit call counters are also restored. Numerical equivalence to uninterrupted training is not claimed. No additional seed or stage1 retraining. Original records and approval are retained in manifests/stage_boundary_recovery.json and the stop archive.

Storage deviation: the user authorized NAS storage after the local filesystem filled during disposable preflight. The entire new task was copied to its private NAS namespace and all 1138 original files/links were verified before replacing the logical output directory with a symlink. The original HF checkpoint retention and stage-endpoint hard links are retained; NAS hard-link/rename/readback checks passed. The absent shared NAS mount is replaced only through this task’s input image aliases to the same NAS objects/local cache; annotations, video/frame selection, geometry, batch and workers are unchanged. Full paths and hashes: manifests/NAS_migration.json and manifests/input_storage_aliases.json. Completed HF50 preflight is reused; the incomplete new-mode attempt is archived and restarted from source S with the same seed. HF preflight logged locally and the restarted candidate logs on NAS, so elapsed and compute timings are both reported; physical GPU IDs are recorded in job manifests. No formal checkpoint recovery occurred during this migration.
