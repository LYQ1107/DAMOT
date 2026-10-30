idAPR gated experiment: idapr_20261030_153059

Stage A completed and FAILED its prespecified method gate. The oracle
regression and artifact checks passed. F failed both identity-recovery
requirements with either stage1 or stage3; HF passed. Nuisance k2 passed
for both models. U checks, training-code modifications and Stage B were
not started, as required. No new training seeds or GPU experiments ran.

Read SUMMARY.md and A/gate.json first. All per-domain results, all ranks
2/8/16, all ten random replicates, and all principal angles are retained.
Recovery ratios are calculated separately for each of the three old
domains and then equally averaged, exactly as requested.

Original CPU commands (no torch/mmcv required by Stage A):
  OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 MKL_NUM_THREADS=2 \
    /data1/liuyeqiang/damot_rebuttal_20261026/env/bin/python \
    tools/analysis/idapr_offline.py --work-dir <W> --model F
  Repeat with --model HF, then run with --aggregate instead of --model.
  Finally run <W>/scripts/validate_A.py in the same Python environment.
The recorded controller ran F and HF independently on CPU in parallel.

The two new analysis modules are published at tools/analysis/ in the
repository and mirrored in this result directory's source/ subtree.
No pre-existing training implementation was modified. To reproduce on
another host, adapt the input locations in manifests/inputs.json and
restore the read-only feature files listed in feature_inputs.json.
The execution controller expects the original <W>/code clone layout;
the direct analysis commands above also work from the repository root.

Raw NPZ features and model weights remain on the server. Small saved
orthogonal .npy bases are included for direct inspection and exact reuse.
The input features were hashed before and after analysis and unchanged.
