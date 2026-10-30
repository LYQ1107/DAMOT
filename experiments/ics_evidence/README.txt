ICS evidence execution and reproduction

Primary run directory: /data1/liuyeqiang/ics_evidence_20261030_215311
Read SUMMARY.md for measured conclusions and all failed prespecified gates.

Environment: /data1/liuyeqiang/damot_rebuttal_20261026/env/bin/python
Python3.7.16; existing torch/mmcv/mmdet and CPU scientific stack. No network
training is performed. Logistic regression probes are fitted on fit videos.

Reproduce part1 on the recorded server inputs:
  OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 MKL_NUM_THREADS=2 CUDA_VISIBLE_DEVICES='' \
    /data1/liuyeqiang/damot_rebuttal_20261026/env/bin/python scripts/evidence_cpu.py --model F
  Repeat for --model HF, then scripts/validate_part1.py and scripts/report.py.

GPU controller: scripts/gpu_controller.py. It uses only physical GPU0/GPU1;
keeps existing allocations; caps each CUDA allocator at35% of total memory;
waits for at least16GB free and low activity before starting an inference.
Before rerunning a controller, inspect manifests/infer_*.json and running
processes. A resumed controller attaches to unfinished existing inferences.
Part3 waits for completed formal C0-C4 training endpoints in the other run.

VKITTI preparation: scripts/prepare_vkitti.py snapshots annotations, samples
every third original frame up to60/video, maximizes valid IoU>=.9 matches
within scene/camera/frame and stages image copies. On these sampled
annotations the extractor stride is1. The initial double-stride attempt was
retained under superseded/double_stride and never used for reported metrics.

Optional native VK model: final proto-shrinkage stage5 after sunset epoch5,
found in the original NAS work_dirs/vkitti2 tree. Its source images may have
been used in its original network training; this row is explicitly separate
from the zero-VKITTI-finetuning BDD-trained F/HF and C-series models.
scripts/native_fit_inputs.py freezes exactly the original HF BDD fit car RoIs
for native-model comparator fitting. No BDD test feature inference is needed.

Offline GPU evaluations: scripts/evaluate_vkitti.py --part 2 or --part 3;
scripts/validate_vkitti.py --part 2 or --part 3; then scripts/report.py.
scripts/analysis_controller.py runs these when each inference marker appears.

The published code/tools/analysis/ics_probe_extract.py is the task-specific
extractor. The local code snapshot includes darth and configs copied from
ics_v2_20261030_184026/code_hf, with source fingerprints in the delivery
manifests. To reproduce elsewhere, restore this matching source tree under
code/, install the dependencies, and adjust read-only input paths. Runtime
C2/C3 fixed-basis paths and checkpoint embedded configurations are retained
by the source checkpoints; no learned tensor is changed for inference.

Data/images, featureNPZs, weights and superseded large feature archives stay
on the server. Publication includes metric CSVs, small basisNPYs, figures,
source, logs, SHA256 input fingerprints and effective inference arguments.
Each part is committed to GitHub experiments/ics_evidence when complete.
