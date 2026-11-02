HF drift-energy supplement: CPU only; immutable v3/A2 bases

Checkout: /data1/liuyeqiang/DAMOT_publish_20261031
Python: /data1/liuyeqiang/damot_rebuttal_20261026/env/bin/python

From the repository root, reproduce analysis and validation:

OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 MKL_NUM_THREADS=4 CUDA_VISIBLE_DEVICES='' bash experiments/ics_figs_v3_energy/scripts/run_all.sh

The same environment is mandatory for each individual command:

OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 MKL_NUM_THREADS=4 CUDA_VISIBLE_DEVICES='' PYTHONDONTWRITEBYTECODE=1 /data1/liuyeqiang/damot_rebuttal_20261026/env/bin/python -B -u experiments/ics_figs_v3_energy/scripts/drift_energy.py
OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 MKL_NUM_THREADS=4 CUDA_VISIBLE_DEVICES='' PYTHONDONTWRITEBYTECODE=1 /data1/liuyeqiang/damot_rebuttal_20261026/env/bin/python -B -u experiments/ics_figs_v3_energy/scripts/validate.py

All output is confined to this new directory. Historical scripts are imported
without executing their command-line main functions. Python bytecode output is
disabled. No training, GPU inference, basis estimation, QR, random draws or
new features are used. Both shell settings and MKL runtime stay at four threads.

The original drift.load reads normalized HF clean features for stages 1-4.
All 20 input NPZ hashes must equal the v3 manifest. Metadata must align exactly
across stages; the video split must equal v3. Car test RoI pairs are:
overcast stage1->4: 1813; rainy stage2->4: 2326; dawn/dusk stage3->4: 3978.
All 8117 pairs enter energy sums, independently of retrieval query limits.
For audit, original retrieval task hashes and counts must also match v3.

Projection matrices are read directly, unchanged, from:
../ics_figs_v3/bases/HF_stage1_FAP_residual_PCA_k{2,8,16}.npy
../ics_figs_v3/bases/HF_stage1_Identity_oracle_k{2,8,16}.npy
../idapr/A/HF/bases/random_k{2,8,16}_rep{0..9}.npy

For d=z_stage4-z_stage_j (each endpoint L2 normalized exactly as v3), energy
is 100*sum(||d@E||^2)/sum(||d||^2). Numerator and denominator are summed over
all RoIs in the three old domains before division. Per-domain rows are also
retained. Neither per-RoI nor per-domain percentages are averaged.

Restoration values are copied directly from
../ics_figs_v3/drift_restoration_HF_native.csv, including the matching random
replicate. The main-table efficiency is recovery_pct/energy_pct; Random uses
ratio of means. Means and SDs of the ten paired replicate efficiencies are
additionally retained in CSV and are explicitly distinct statistics.

Random SD is population SD (ddof=0), matching v3. Theory k/256 is 0.78125%,
3.125%, 6.25%. The predeclared broad sanity diagnostic asks whether theory
lies in the empirical random mean +/-2SD; it is not a significance test and
does not trigger re-estimation, replacement or filtering of random bases.
Exact means, SDs, min/max and differences from theory are always published.

drift_energy_HF_native.csv contains 156 rows:
  domain: 27 rows, including 9 random means.
  summary: 9 three-domain ratios, including 3 random means.
  random_replicate_domain: 90 rows.
  random_replicate_summary: 30 rows.
Energy and recovery columns are percentages; projected_energy_sum and
total_drift_energy_sum are raw squared L2 feature units; efficiency is their
percentage-ratio quotient. No clipping, sign filtering or zero-drift exclusion
is performed. Source fingerprints, cohorts, raw summary data and validation
results are delivered alongside the report and execution logs.
