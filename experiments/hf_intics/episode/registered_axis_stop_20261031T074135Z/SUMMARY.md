# HF + online intervention-residual ICS

Status: **awaiting angle-definition clarification; stopped under the initially registered PC1/PC1 rule**. Single training seed **1790355234**, identical to HF.

Baseline: /data1/liuyeqiang/damot_apr_complement_hf_20261028_1420. Clear source S → Overcast → Rainy → Dawn/Dusk → Night, five epochs/domain, four V100s, FP32, 16 samples/GPU, two workers/GPU. Geometry, student low-frequency transform, contrastive high-frequency noise, APR and beta=1 coupling are retained. Only the requested ICS settings change; config_diff.json lists every output-path and model difference.

Online update: normalized student key-positive embedding minus normalized clean student embedding at exactly the same boxes; detached residuals, no mean subtraction or residual renormalization, covariance EMA .98, shrinkage0, two PCs, HF PC EMA .9 followed by QR, alpha .25, loss weight .1, min_samples32. The APR basis reader and trim operator are unchanged.

Preflight:

The 45° condition requires clarification before further training or final publication. The initially registered comparison is bank PC1/offline PC1 (82.480923°), which stopped the process. The principal angle from bank PC1 to the full offline2D subspace is33.756089°; both values were independently verified with scipy.linalg.subspace_angles. The two2D principal angles are33.453465°/88.357491°. See manifests/angle_clarification.json and probe/stage1/angle_definition_check.json. [SciPy defines the comparison in terms of the input column spaces](https://docs.scipy.org/doc/scipy/reference/generated/scipy.linalg.subspace_angles.html). All first-stage results and saved states remain available; no post-stage1 training or final publication has occurred.

- U1: passed; total and all component losses bitwise equal, 0.7292587757110596.
- U2: principal angles 65.609289°, 74.419794°; reported without an angle threshold.
- U3: passed; paired disposable HF/new50-iteration runs, iteration-time increase 0.8468%.
  New key/ref actual nonidentity rates 99.5000%/100.0000%; mean weighted loss_track_trim 0.077684235; first initialization by rank [1, 1, 1, 1].
  Mean critical-rank iteration time on iterations11–50: HF 22.212150s, INTICS 22.400244s.
  Training-step timing excluding batch fetch and the independent CSV write: HF 22.112865s, INTICS 22.050363s, signed change -0.2826%. HF logs were local and candidate logs were on NAS; these are single paired timing measurements.

The original HF metadata uses cumulative teacher scale_factor and local student scale_factor=1; geometry assertions compare shapes, flips, exact box coordinates and the equivalent composed scale, while retaining the original metadata. Both views remain on identical pixels/geometry. Fixed-batch cache is disposable and stored in this task’s /dev/shm cache symlink.

Stage1 gate: OVERALL HOTA delta <=−0.5, or acute bank PC1/offline FAP PC1 angle >45°. Offline directions use only fit videos, normalized paired FAP/clean residuals without centering. The process waits at the stage boundary with model/RNG state retained; gate inference may share only its own paused training cards.

Stage1 decision: **STOP**; bank PC1/offline FAP PC1 angle exceeds45deg.
Overcast OVERALL HOTA 36.474774 vs HF 36.781975 (delta -0.307201). Bank PC1/offline PC1 angle 82.480923°; PC1/offline2D angle 33.756089°; two subspace principal angles [33.453465219188544, 88.357491312671].
Raw decision: episode/stage1_gate.json.

Stage1 P1 energies use test videos; offline FAP directions use fit videos only.

| Bank directions | Cross-domain shift energy | Identity energy |
|---|---:|---:|
| Both PCs | 8.5078% | 9.1129% |
| PC1 | 7.1322% | 5.1749% |
| PC2 | 1.3756% | 3.9379% |

| Stage | Domain | Category | HOTA | DetA | IDF1 | AssA | ΔHOTA vs HF |
|---:|---|---|---:|---:|---:|---:|---:|
| 1 | overcast | car | 40.115006 | 31.828057 | 46.316529 | 50.727402 | -0.331385 |
| 1 | overcast | OVERALL | 36.474774 | 27.820921 | 40.979041 | 48.055223 | -0.307201 |

Missing later stages/cells have no result; four-domain final/AIP/AvgF/FinalOld/BWT are not fabricated.

Probe results: probe/stage1/P1.json, and (if stage4 is reached) probe/stage4/P1.json plus probe/stage4/evidence/part1/INTICS/ for Table1 retrieval, Table2b deletion and FigureC crops. Bank diagnostics read the rank0 student bank from the frozen checkpoint; all four rank-local runtime states are retained as in HF. The actual bank_pcs.npy and offline_fap_k2.npy with SHA256 hashes accompany each P1 report. Random-basis probe controls use the historical fixed seed0; they are not additional training seeds.

No epoch-boundary recovery is recorded. Requested ICS changes, geometric-metadata equivalence checks, disposable preflight weights and the stage1 pause are disclosed above; all other HF training settings remain unchanged.

Storage deviation: the user authorized NAS storage after the local filesystem filled during disposable preflight. The entire new task was copied to its private NAS namespace and all 1138 original files/links were verified before replacing the logical output directory with a symlink. The original HF checkpoint retention and stage-endpoint hard links are retained; NAS hard-link/rename/readback checks passed. The absent shared NAS mount is replaced only through this task’s input image aliases to the same NAS objects/local cache; annotations, video/frame selection, geometry, batch and workers are unchanged. Full paths and hashes: manifests/NAS_migration.json and manifests/input_storage_aliases.json. Completed HF50 preflight is reused; the incomplete new-mode attempt is archived and restarted from source S with the same seed. HF preflight logged locally and the restarted candidate logs on NAS, so elapsed and compute timings are both reported; physical GPU IDs are recorded in job manifests. No formal checkpoint recovery occurred during this migration.
