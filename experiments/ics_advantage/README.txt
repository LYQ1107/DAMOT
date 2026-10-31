ICS advantage evaluation (one trained seed, no network training)
Run root: /data1/liuyeqiang/ics_advantage_20261031_004629
Python: /data1/liuyeqiang/damot_rebuttal_20261026/env/bin/python
Read SUMMARY.md for every result, uncertainty and implementation choice.
All task outputs are isolated here; other run directories are read-only.

Reproduce prepared inputs: scripts/prepare_inputs.py. Frozen historical test split
seed0, original stride2 first30 frames/video, all GT cars, first5 dose frames.
The preselected annotations are inferred with stride1 (never stride2 again).

Inference queue: scripts/gpu_controller.py. Inspect manifests/infer_*.json and
live PIDs before resuming; running children are reattached, never duplicated.
Only fully idle GPUs are used under the original current-task policy. Shared
GPU0/1 requires the separate direct human resource-policy approval recorded
in manifests/resource_policy.json; absence of that file means no sharing.
All other training allocations are preserved. MKL_THREADING_LAYER=GNU avoids
an observed OpenMP initialization failure. Its failed log is retained, and only
the failed small supplemental extraction was retried. C3 continued unchanged.

CPU sequence evaluation: scripts/tracking_per_video.py --model C0 --domain night
(repeat five models and two domains, or use scripts/cpu_controller.py).
TrackEval frozen detailed threshold arrays supply HOTA/AssA. Original native
MMTracking/MOTMetrics supplies IDF1/IDSW. All80 aggregate parity checks pass.
Formal paired comparisons use test videos only; original aggregates use all
videos for protocol parity. scripts/brightness.py reads every original frame.
scripts/tracking_statistics.py --batch 1 produces paired bootstrap10000 seed0,
two-sided Wilcoxon, fixed brightness thirds and Spearman results.

Feature analysis: scripts/feature_statistics.py --batch 1, then
scripts/feature_figures.py --batch 1. P2 uses the original temporal window and
max4000 seed0; sparse doses have explicit matched query/gallery clean baselines.
Cosine silhouette is exact over all available requested view samples; the
cluster-mean implementation matches sklearn in scripts/selftest.py.
No inference-time ICS projection is applied to association embeddings.

Supplemental t-SNE views are extracted separately to provide the three requested
doses at the four uniform trajectory frames. These features are excluded from
all primary quantitative statistics. All selections are frozen in manifests.

Second-batch controller: scripts/batch2_controller.py. It waited for both the
first-batch publication receipt and the independent source training/frozen
evaluations. It copied the final source to code_online and overlaid the identical
task extractor, then evaluated HF/new through that same inference implementation.
Source model training was never launched or modified here. The stage4 epoch5
source checkpoint is now available and the four-domain stage4 feature, tracking,
VKITTI2 and offline analyses are complete. `manifests/batch2_validation.json`
records PASS (all four models, 18 figures, 80 native aggregate checks and valid
10,000-repetition bootstrap intervals); the stage1 supplement is retained.

All data images, model weights, frozen predictions and feature NPZs remain on
the server. Delivery includes full metric CSVs, PDF/PNG figures, exact source
snapshots, selection/protocol metadata, versions, hashes and logs. Public source
checkpoint paths and constructor basis paths must be remapped when reproducing
on another machine. Small fixed constructor bases are included.

Before publishing, run scripts/report.py, scripts/validate_delivery.py --batch 2,
then scripts/build_delivery.py; verify SHA256SUMS from delivery/. The published
tree contains both batches, the stage4 supplement, validation manifests and the
exact code snapshot used for extraction and analysis. Public source checkpoint
paths and constructor basis paths must be remapped when reproducing on another
machine.
