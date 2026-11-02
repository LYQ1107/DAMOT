HF-native ICS figures v3: CPU only, no training or feature extraction

Checkout: /data1/liuyeqiang/DAMOT_publish_20261031
Python: /data1/liuyeqiang/damot_rebuttal_20261026/env/bin/python (Python 3.7)
All outputs, logs and matplotlib cache stay under experiments/ics_figs_v3/.
Historical experiment files, image data and NPZ features are read-only.

From the repository root, reproduce all three tasks:

OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 MKL_NUM_THREADS=4 CUDA_VISIBLE_DEVICES='' bash experiments/ics_figs_v3/scripts/run_all.sh

Individual commands, in dependency order:

OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 MKL_NUM_THREADS=4 CUDA_VISIBLE_DEVICES='' PYTHONDONTWRITEBYTECODE=1 /data1/liuyeqiang/damot_rebuttal_20261026/env/bin/python -B -u experiments/ics_figs_v3/scripts/hf_native.py
OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 MKL_NUM_THREADS=4 CUDA_VISIBLE_DEVICES='' PYTHONDONTWRITEBYTECODE=1 /data1/liuyeqiang/damot_rebuttal_20261026/env/bin/python -B -u experiments/ics_figs_v3/scripts/figures.py
OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 MKL_NUM_THREADS=4 CUDA_VISIBLE_DEVICES='' PYTHONDONTWRITEBYTECODE=1 /data1/liuyeqiang/damot_rebuttal_20261026/env/bin/python -B -u experiments/ics_figs_v3/scripts/report.py
OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 MKL_NUM_THREADS=4 CUDA_VISIBLE_DEVICES='' PYTHONDONTWRITEBYTECODE=1 /data1/liuyeqiang/damot_rebuttal_20261026/env/bin/python -B -u experiments/ics_figs_v3/scripts/validate.py

figures.py --which drift or --which crops runs only that plotting component.
report.py uses the retained visual_review.json, which records the direct review
of delivered figures and never controls the selected data or computed metrics.
Initial syntax failure, corrected before analysis, is retained in logs/ for audit.

Inputs:
- HF_feat_root in ../idapr/inputs.txt; all five domains of HF_stage1_e5 are
  explicitly checked for finite feat_clean and feat_fap before computation.
  Missing HF inputs cause immediate failure; no F feature fallback exists.
- Task 1 loads HF stages 1/2/3/4 with unmodified ics_probe_drift.load. It calls
  the original idapr_offline.oracle_bases with HF stage1 reference data.
  FAP residual PCA is centered PCA of normalize(feat_fap)-Z on fit videos.
  Identity oracle is uncentered PCA of identity_rows on the same fit videos.
- Label-free nuisance and all random replicates are reused from historical
  A2 HF rows as requested. Their input CSVs remain unchanged.
- Task 2 reads the fixed twenty rows, row/column order and floating crop boxes
  in ../ics_evidence/part1/HF/plot_C_crops.csv. Only the global title and small
  labels change; source images and selected RoIs are not replaced.
- Task 3 directly reads INT_00.npy. The label-free identity basis is recomputed
  with the original label_free_bases on HF stage4 fit car RoIs, paired
  FAP/photo/student views, eps_ratio=0.1, image IDs namespaced on collision.
  Identity/weather labels enter only the test-energy metrics, not this fit.

Imports resolve to the immutable idapr/source/tools/analysis, ics_evidence/scripts
and tools/analysis modules; exact paths and SHA-256 fingerprints are recorded.
No historical scripts are run through their command-line entry points.
Thread environment AND MKL runtime remain at four; no historical two-thread
compatibility context or learned classification probe is needed in this task.

CSV conventions:
drift_restoration_HF_native.csv contains 168 rows:
  domain: 36 rows = 3 old domains x 3 ranks x 4 bases (Random is the mean of 10).
  summary: 12 rows = 3 ranks x 4 bases, using ratio of sums over old domains.
  random_replicate_domain: 90 rows = 3 domains x 3 ranks x 10 fixed replicates.
  random_replicate_summary: 30 rows = 3 ranks x 10 replicate ratios of sums.
For each summary, recovered_pct = 100*sum(reduction)/sum(proxy), residual_pp =
100*sum(restored)/3 and none_pp = 100*sum(proxy)/3. Raw forgetting columns in
summary rows are sums, while mAP columns are intentionally blank. Domain rows
contain per-domain values. Raw mAP is 0-1; *_pct and *_pp use percent/pp.
Random SD is the population SD (ddof=0) of ten replicate recovery percentages;
min/max use those same ten values. These are not training-seed confidence intervals.

S1_energy_check.csv contains the direct INT result, recomputed label-free identity
result, and both saved A1 bases as controls. It retains exact differences and
principal angles. A difference is reported without overwriting INT or A1 results.
Real cross-weather energy uses A1's weighted domain-mean shifts against Clear,
not all weather pairs or the average of per-RoI projected fractions.

The six PDF/PNG files are scientific plots from matplotlib/Pillow. Drift width is
exactly 3.3 inches, fonts >=8pt and PNG resolution 300dpi; the fixed 2x10 crop
figure preserves the original 14x3.8-inch layout without a global title.
Feature arrays and source images remain in their existing server locations.
