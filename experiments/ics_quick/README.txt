ICS mechanism experiment run: ics_quick_20261029_160807

Status and results: SUMMARY.md; manifests/result_status.json.
The run was prepared on 2026-10-29 in Asia/Shanghai time.
The input directories, checkpoints and source experiment are read-only.

Reproduction inputs are enumerated in inputs.txt and manifests/inputs.json.
F code is a fresh clone checked out at ebc3190b2fa82d9e117315998c0d7435033e113c.
HF code is copied from the actual archived run, rather than GitHub main.
Local HF edits are in manifests/HF_code_changes.patch; new modules are
preserved under manifests/source_additions/. No shared Python package is edited.
The independent NAS connection is read-only and its data_views resolve to
the same original images; see manifests/private_view_byte_check.json.

Environment:
  /data1/liuyeqiang/damot_rebuttal_20261026/env/bin/python

Core commands (run in this run's original workspace, or reconstruct the
code/code_hf snapshots and adjust input locations on another machine):

1. X1: run scripts/grad_compare.py --model F|HF --domain night|overcast.
   Each invocation fixes the archived seed and takes exactly 20 batch16
   iterations from the original target training pipeline, without optimization.
   Non-embedding parameters do not retain unused gradient graphs; all model
   weights and forward computations remain those of the full adaptation model.

2. X2: extract HF clear_holdout_45v using the original ics_probe_extract.py
   with --views clean --dose-frames-per-video 0 and all other default sampling
   options; the exact initial command is in manifests/X2_HF_extract.json.
   Run scripts/basis.py --model F and --model HF, then scripts/evaluate.py
   --model F|HF --domain night|overcast --beta 0 --check-disabled.
   A failed frozen baseline assertion blocks beta1. Once beta0 passes, run
   the same cell with --beta 1. Completed video predictions are reused when
   a transient input connection failure is retried.

3. X3: scripts/prepare_x3.py creates the four base configs and copies the
   fixed stage3 checkpoint. Shared original APR prototypes come from the
   stage3 rank0 archive; their source, shapes and SHA256 are recorded in
   X3/apr_history_manifest.json. Each group uses the resolved parameters in
   X3/<group>/resolved_stage4.py and the exact distributed launch command
   recorded by scripts/gpu_job.py in manifests/<group>_train_job.json.
   The trainer checks that three history domains are restored before training.
   Audit files count trim activation and bank updates separately for each rank.

scripts/gpu_job.py rechecks nvidia-smi before each launch and waits for the
full requested number of wholly idle GPUs. It never kills another process.
scripts/controller.py records orchestration for this live run; its adopted
PIDs are session-specific and should not be reused for a fresh reproduction.
For fresh reproduction, use the core commands and the recorded final commands.

Checkpoints, feature NPZ files, raw prediction pickle files and the private
mount are server-only. The publication bundle contains CSV results, bases,
configs, logs, scripts, patches, manifests and SUMMARY.md.
