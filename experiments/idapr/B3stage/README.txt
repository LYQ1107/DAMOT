B3 disturbance-release follow-up, authorized after the original idAPR A gate.
This task does not run B1/B2, tune the failed identity estimator, or add seeds.

Read SUMMARY.md for the single requested table. HOTA/DetA/IDF1/AssA and all
forgetting values are percentage points. Old-domain forgetting is HF stage3
minus the final value; the mean equally weights Overcast, Rainy, Dawn/Dusk.
Negative forgetting is retained. Comparisons within +/-0.3 HOTA are labeled
no difference, as requested. No statistical significance is inferred.

Formal training: one seed1790355234, two Night epochs, four V100s per group,
samples16/GPU and workers2/GPU. B3 releases a fixed teacher-derived two-space
and disables APR-ICS coupling; R retains original HF coupling beta1 / ICS /
projected_cosine and uses the default full mode. Both retain the native ICS.
B0 reuses X3-0; only its missing Overcast/Rainy evaluations are added.

Quick checks:
  U/U1.json                 CPU torch loss and normalized-node gradient checks
  U/U2.json                 same-batch original/full losses, bitwise equal
  U/extractor_parity.json   default GT arrays byte-identical; teacher has no GT
  U/U3.json                 exactly50 smoke updates, finite nonzero APR, history3
  U/warmstart.json          strict full student+teacher state and shared history
  U/all_passed.json         gate recorded before either formal group starts

U3 does not retrain B0. The archive contains20-step averages, so its available
first40 steps are an unpaired native reference. A paired full-space loss is
also computed on each B3 step's exact same features/prototypes/gate, giving
50 matching reference values. Both ratios and this scope are reported.
Smoke weights are discarded; formal runs independently restart at stage3.

Basis: all33 Dawn/Dusk videos, original sampling30frames/video at stride2,
990 planned frames, teacher car score>=.75, clean/FAP/photo/student views.
2851 car RoIs across870 images; all IDs=-1. No GT annotations enter selection.
The basis provenance, SHA256 and angles to the prior A fit are in basis/.

Execution uses an isolated copy of the actual X3 HF runtime, including its
non-reentrant checkpointing, restored APR history and local audit hooks.
The old task, original HF code, input NPZs and weights are read-only.
The changed modules are mirrored in source/, with original comparison files
in source_original/. code_changes.patch applies to the recorded X3 HF copy.
No root training modules elsewhere in the repository are overwritten.

Use inputs.txt and manifests/*_job.json for exact paths and launch arguments.
scripts/controller.py gates the two formal runs and requests four wholly idle
GPUs each; it permits two concurrent groups when eight GPUs are idle. It does
not kill other processes or reduce cards/batch. Stage3 teacher/student tensors
and the identical a8c2ea74... APR-history bytes are restored in every rank.

Raw NPZs, model weights, raw per-frame predictions and the private NAS mount
remain server-only. The private connection is read-only and shares identical
source image bytes without changing any existing mount or other task files.

Checkpoint serialization compatibility: old MMCV ignores persistent=False. The final code_changes.patch adds a release-only exclusion hook. code_training.patch and source_training/ preserve the exact formal training source. Original checkpoints and evaluation provenance remain unchanged. B3 epoch_2_export.pth removes only the two fixed external basis buffers; U/checkpoint_serialization.json proves all 752 retained tensors and all other payload bitwise identical. Original per-rank audit basis_not_persisted_in_checkpoint flags checked only PyTorch state_dict; training_audit.csv labels this as an original hook claim and reports actual MMCV-file findings separately. No training or inference was repeated.
