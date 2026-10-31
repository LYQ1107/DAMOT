HF + online intervention-residual ICS, paired seed1790355234.

Read SUMMARY.md for measured preflight/gate/results; pending cells have no value.
Code is an isolated copy of the actual HF runtime:
  /data1/liuyeqiang/damot_apr_complement_hf_20261028_1420/code
Original HF, other tasks, checkpoints, annotations and image views are read-only.

Runtime:
  /data1/liuyeqiang/damot_rebuttal_20261026/env/bin/python
Python3.7 model stack; system Python3.9 for controllers/reporting.
Four V100s, FP32, batch16/GPU, workers2/GPU, persistent workers as HF,
same geometry, student low-frequency, contrastive high-frequency and beta1 APR.

Only ICS changes: intervention_residual,2PCs,alpha.25,weight.1,min32,shrink0;
covEMA.98 and PCEMA.9 retained, with QR after PC mixing for this mode.
Residuals are normalized-student minus normalized-clean at identical key
positive RoIs. No residual re-normalization, no mean subtraction; the original
HF N-1 covariance denominator remains. APR's basis reader is unchanged.

scripts/preflight_batch.py compares original vs disabled new code on exactly
one saved Dawn/Dusk batch with real HF stage3 APR history, then reports online
residual top2 angles to the A-stage HF3 FAP nuisance2 basis. U2 has no angle gate.
Raw HF scale_factor metadata differs by reference frame (cumulative teacher,
local student1). Pixel shapes, flips, box coordinates and composed scales are
asserted equivalent; HF metadata is retained.
scripts/source_checks.py checks unchanged APR/trim ASTs, default covariance
bit equality, uncentered moment, QR and finite trim gradients.

scripts/smoke_controller.py runs disposable HF and INTICS 50-step smoke runs
using the same four-card HF runner/seed/batches. It checks per-rank key/ref
nonidentity rates >=80%, records first bank init and mean loss_track_trim.
Timing uses maximum rank duration per iteration, then averages iterations11-50.
Neither smoke's weights are used by formal training.

scripts/formal_controller.py waits for all preflight checks to pass and for
four wholly idle cards. Formal training starts from the original Clear S and
uses the original HF stage runner. At the end of stage1 it holds the live
model/RNG state, evaluates Overcast and runs clean/FAP probes on its own
paused cards. STOP if OVERALL HOTA is at least.5 below HF R11 or the principal
angle between bank PC1 and the offline two-dimensional FAP subspace exceeds45
degrees. PC1-to-PC1 and both2D principal angles are also retained as diagnostics.
The initial registration used PC1-to-PC1(82.480923deg) and stopped after stage1.
The user explicitly clarified the condition to PC1-to-offline2D(33.756089deg)
and authorized continuation from the full stage1 boundary. Original records
are preserved under the archive recorded in manifests/angle_clarification.json.
The same seed continues without stage1 retraining or model/config changes.
scripts/train_with_resume_audit.py calls the unchanged HF restore and verifies
all parameters, rank buffers, APR runtime state, counters and parent RNGs before
stage2. FAP telemetry counts are restored too. Each new domain creates fresh
workers and an optimizer as in HF; uninterrupted numerical equivalence is not
claimed. See manifests/stage_boundary_recovery.json and per-rank restore audits.
The user authorized NAS storage after local space filled during disposable
preflight. All1138 existing task files/links were verified after bulk copying
to the private NAS namespace; the original logical output path is a symlink.
Original HF checkpoint retention and stage endpoint hard links are unchanged.
Hard-link/rename/readback checks passed on NAS; formal launch requires8GiB free
on NAS and retains the original HF1.5GiB post-save guard on the output filesystem.
This task uses independent image aliases to the same original NAS objects and
local cache because the shared NAS mount is absent. Old task files are untouched.
Details: manifests/NAS_migration.json and manifests/input_storage_aliases.json.
Completed HF50 smoke is reused; incomplete INTICS smoke was archived and is
restarted from S with the same seed. HF50 logged locally, restarted INTICS50
logs on NAS; both elapsed and compute timing and physical GPU IDs are retained.

scripts/stage_probe.py preserves original P1 canonical GT-box sampling:
car, fixed geometry1296x720, max40videos/domain,30frames/video,stride2,
min16px,max40RoIs/frame, perturbation seed20261028. P1 uses test videos,
Clear-referenced weather shifts and video-relative identity rows. Offline
intervention direction uses uncentered FAP residuals on fit videos only.
Probe split seed0 is separate from the single network training seed.
At stage4 the original evidence part1 code adds Table1 cross-view retrieval,
Table2b deletion and FigureC extrema crops, retaining its historical controls.

scripts/report.py compares every measured cell to HF and computes final R4j,
diagonal Rjj, AIP, AvgF, FinalOld, BWT for OVERALL and car without clipping.
If the stage1 gate stops the run, later cells are missing, not zeros.
scripts/validate_completed.py validates source seal, preflight, mandatory gate,
stage checkpoints and full frozen evaluation pools.

Publication: experiments/hf_intics/ only. Model weights, cached image batches,
raw prediction PKLs and feature NPZs stay on the server. Changed source files,
the exact baseline files, a patch, configurations, CSVs, figures and audits are
included. Repeated per-frame log progress prefixes are compacted only in the
published copy; original log hashes and original logs remain on the server.
