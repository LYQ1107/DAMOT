# Rebuttal runner scripts

These are the scripts used for the V100 run. They are an execution snapshot,
not a claim that the public repository contains the private BDD images or
checkpoint weights. Prepare the same directory layout (`results/`,
`manifests/`, `configs/`, `cache/weights/`, and `cache/images/`) and install
the compatible MMCV/MMDetection/MMTracking stack before running them.

The executed order was:

```text
collect_exp1.py
evaluate_clear.py --model S
evaluate_clear.py --model F_stage1_e5
evaluate_clear.py --model F_stage2_e5
evaluate_clear.py --model F_stage3_e5
evaluate_clear.py --model F_stage4_e5
diagnose_teacher.py --timepoint e0
diagnose_teacher.py --timepoint e1
diagnose_teacher.py --timepoint e3
diagnose_teacher.py --timepoint e5
profile_damot.py rebuild
profile_damot.py throughput
profile_damot.py modules --history 0
profile_damot.py modules --history 1
profile_damot.py modules --history 2
profile_damot.py modules --history 3
profile_damot.py inference
aggregate_results.py all
validate_delivery.py
```

`retest_fap_ics_interleaved.py` is the later GPU-1 retest. It alternates the
two groups in one process and writes to
`results/exp4/retest_fap_ics_20261027/` without overwriting the archived main
table. All formal windows were FP32, CUDA-synchronized at boundaries, and
used the same fixed frame stream and batch size.

Private NAS transfer helpers are intentionally not included. The scripts
expect data and weights to be staged locally; `reproduce.sh` is therefore the
as-run controller and requires those private staging steps before it can be
used. No GitHub credential, SSH key, or private mount is required by the
repository itself.
