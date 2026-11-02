ICS figures v2: offline CPU-only reproduction

Repository checkout: /data1/liuyeqiang/DAMOT_publish_20261031
Python: /data1/liuyeqiang/damot_rebuttal_20261026/env/bin/python
All inputs are read-only. Every output, log and cache is under experiments/ics_figs_v2.
No network training, new feature extraction, GPU use or basis estimation occurs.

From the checkout root, reproduce the complete package with:

OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 MKL_NUM_THREADS=4 CUDA_VISIBLE_DEVICES='' bash experiments/ics_figs_v2/scripts/run_all.sh

To run an individual component, use the same environment prefix:

OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 MKL_NUM_THREADS=4 CUDA_VISIBLE_DEVICES='' PYTHONDONTWRITEBYTECODE=1 /data1/liuyeqiang/damot_rebuttal_20261026/env/bin/python -B -u experiments/ics_figs_v2/scripts/drift_figures.py
OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 MKL_NUM_THREADS=4 CUDA_VISIBLE_DEVICES='' PYTHONDONTWRITEBYTECODE=1 /data1/liuyeqiang/damot_rebuttal_20261026/env/bin/python -B -u experiments/ics_figs_v2/scripts/quadrant_analysis.py --model F
OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 MKL_NUM_THREADS=4 CUDA_VISIBLE_DEVICES='' PYTHONDONTWRITEBYTECODE=1 /data1/liuyeqiang/damot_rebuttal_20261026/env/bin/python -B -u experiments/ics_figs_v2/scripts/quadrant_analysis.py --model HF
OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 MKL_NUM_THREADS=4 CUDA_VISIBLE_DEVICES='' PYTHONDONTWRITEBYTECODE=1 /data1/liuyeqiang/damot_rebuttal_20261026/env/bin/python -B -u experiments/ics_figs_v2/scripts/build_report.py
OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 MKL_NUM_THREADS=4 CUDA_VISIBLE_DEVICES='' PYTHONDONTWRITEBYTECODE=1 /data1/liuyeqiang/damot_rebuttal_20261026/env/bin/python -B -u experiments/ics_figs_v2/scripts/validate.py

quadrant_analysis.py --phase metrics computes only metrics; --phase figures reads
the completed metric CSV and regenerates figures. The default all runs both.
build_report.py also requires the retained visual_review.json; it records the
inspection of the delivered figures, not a metric-dependent sampling decision.

The prescribed four-thread shell environment is checked by common.py. Historical
weather-probe scores require the original two-thread MKL numerical path: the
original command is recorded in ../ics_evidence/logs/part1_F_terminal_transcript.txt.
common.historical_probe_blas temporarily sets MKL=2 for historical load/projection/
logistic-probe operations and restores MKL=4 afterward. The unchanged historical
weather_probe is still called directly; PCA, t-SNE, exact silhouettes and other
operations use four threads. The initial failed four-thread selfchecks and the
1/2/4-thread diagnostic remain in logs/. No tolerance was relaxed to pass.

Metrics use all 18,019 test car RoIs per model. The only query cap is the original
build_retrieval(..., gap=4, window=30, max_q=4000, seed=0). Gallery candidates use
the full test set. Silhouettes use every eligible RoI within every eligible video,
not a 4,000-point display subsample. Within-identity variance uses all test RoIs
in each nonconstant video, including short identities. All per-video values are
weighted by their relevant RoI count. n_videos_used refers to silhouette videos;
the separate variance counts remove ambiguity.

mAP and balanced accuracy are percentages (0 to 100) in quadrant CSVs; silhouette
and within_var_ratio are unscaled. Random mean/SD always includes all ten fixed
bases; SD uses ddof=0. The visual Random column is always R_00.

Drift recovery is a ratio of sums over the three old domains, not the mean of
domain ratios. Random recovery is calculated independently for each replicate
before its mean/SD/min/max are reported. The B-version error bars are the SD of
replicate residual losses, not the SD of recovery percentages.

Drift A/B/C are each delivered at exact widths 3.3 and 7 inches in PDF and 300dpi
PNG. Quadrants are 7 inches wide, 300dpi PNG/PDF, fonts at least 9pt. PCA and t-SNE
use the exact original display indices and highlighted identities, with all-test
metrics shown before dimensionality reduction. Every space uses the same method
within a figure; complement t-SNE uses PCA50 first, exactly as requested.

Original large NPZ features remain in their historical server locations, resolved
by unmodified evidence_cpu.load. No checkpoints, images or full 256D features are
copied into this package. Input fingerprints are retained in protocol JSONs.
