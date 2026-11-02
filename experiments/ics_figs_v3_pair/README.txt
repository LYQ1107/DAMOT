HF paired drift-share and restoration figures (CSV-only CPU plotting)

From /data1/liuyeqiang/DAMOT_publish_20261031:

OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 MKL_NUM_THREADS=4 CUDA_VISIBLE_DEVICES='' PYTHONDONTWRITEBYTECODE=1 /data1/liuyeqiang/damot_rebuttal_20261026/env/bin/python -B -u experiments/ics_figs_v3_pair/scripts/plot_pair.py

Only two frozen CSVs are read: ../ics_figs_v3_energy/energy_summary.csv and
../ics_figs_v3/drift_summary.csv. Values and random SD are not recalculated.
The latter is a regression check for every plotted restoration value.
No features, checkpoints, GPU or training are needed. No existing experiment
directory is modified. Source hashes and plot specifications are recorded.

Outputs, each as vector PDF and 300dpi PNG:
- drift_share_HF_3bars: standalone companion, 3.3 x 3.5 inches.
- drift_pair_HF_horizontal: compact two-panel single-column figure, 3.3 x 2.45.
- drift_pair_HF_vertical: taller single-column alternative, 3.3 x 4.6.

All fonts are at least 8pt. Colors, ranks, fixed bases and k=2 shading match
the previous HF three-bar plot. Paired figures share one legend. The panels
have distinct y-axis ranges because drift energy and restoration are different
quantities. Random error bars are the SD over ten fixed random bases (ddof=0).
The compact layout puts metric names above the panels to preserve plot width.

Drift share = sum ||(z_stage4-z_stage_j) E||^2 / sum ||z_stage4-z_stage_j||^2,
with L2-normalized clean features, pooled over the same 8117 test car RoI pairs
in overcast (stage1), rainy (stage2), dawn/dusk (stage3). Bases were fitted on
HF stage1 in prior work; they are not refitted here. Recovered forgetting is
the restoration of the cross-stage identity-retrieval loss proxy from v3.
