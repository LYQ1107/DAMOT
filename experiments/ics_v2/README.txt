ICS v2: five APR-off Night experiments and twelve historical inference cells

Final specification: C0-C4 all use HF stage3 learned weights, Night only, two epochs,
four V100s/group, seed1790355234, 16 samples and two workers/GPU, min_samples32,
use_roi_proto_memory=False, roi_proto_couple_ics=False. No APR history is loaded.
C0 has no ICS; C1 estimates old global_pc_ema/one PC from a fresh bank; C2/C3
freeze A/HF stage3 FAP/GT/fit nuisance-2 at alpha.25/.5; C4 freezes a seed0 random
orthogonal2 at alpha.25. C1-C4 loss weight.1. See runs/*/config_diff.json.

The first historical table was reported before training. Additional inference
was then authorized for two existing no-ICS runs: stage2 overcast; stage3
overcast/rainy; stage4 overcast/rainy/dawn. No historical training is repeated.
Historical metrics use the original legacy evaluator, native four-GPU test.py,
cuDNN benchmark=True, six workers/GPU, fusion, init threshold.30. New C-series
metrics retain X3's all-frame HOTA, native frozen student and thresholds.05/.30/.20.
These two evaluator variants are kept separate. Historical runs have different
seeds and APR strengths/batches; the comparison is descriptive.

Reproduction on the original server:
1. Use /data1/liuyeqiang/ics_quick_20261029_160807/code_hf as the patch baseline.
   manifests/runtime_baseline.json records its source hashes and archived GitHub paths.
   Apply code_changes.patch there. source/ mirrors every sealed critical module;
   source_original/ contains baseline versions of changed existing modules.
2. Keep dataset views read-only. Input weights, annotations and original runtime
   paths are recorded in inputs.txt and manifests/inputs.json. Starting stage3
   snapshots are read-only symlinks, with SHA256 checked before each run.
3. prepare_groups.py builds the final five configs. preflight.py verifies strict
   learned/nonbank restoration, C1 bank-only reset, fixed-basis geometry and the
   trim operator on CPU. No extra optimizer/smoke-training run is performed.
4. historical_inference.py performs only the twelve authorized missing cells.
   summarize_historical_completed.py merges them with archived metrics.
5. controller.py waits for historical inference, then runs five groups with up
   to two simultaneous four-GPU jobs. gpu_job.py requires wholly idle GPUs and
   preserves prelaunch snapshots. No APR-history argument is supplied.
6. Key and ref nonidentity activity is measured at each actual bank.trim call;
   first50 audits record the 80% condition. C1's first initialization iteration
   and weighted loss_track_trim means are recorded for every rank.
7. Each new model is evaluated on full Night and Dawn/Dusk. IDSW comes from
   the original CLEAR track_metrics.csv OVERALL row, not a class average.
8. validate_completed.py checks all artifacts; summarize.py writes the main
   table, criteria, training_audit.csv and activation_first50.csv.

Only this task's redundant epoch1/stage00 files and duplicate COCO exports are
removed. Epoch1 recovery checkpoints remain until training succeeds; unused
stage00 copies may be removed earlier to reserve disk space. Final epoch2 and
the endpoint remain on the server. Cleanup manifests record removed hashes. Weights, raw
predictions, feature NPZs and data views are excluded from the GitHub package.
X3-0/X3-1 are reused APR-on references, excluded from the revised criterion.
Only seed1790355234 is used for new training. Random-basis seed0 is fixed once
and is not a second training seed. C1/C2/C3 qualify if Night HOTA or AssA is at
least0.4 above C0; C2/C3 must also exceed C4 in the same metric. This is a
single-seed criterion, not a statistical significance claim.

Published inference logs omit repeated carriage-return frame progress prefixes;
all other messages, metrics, configurations and numeric training logs remain.
manifests/publication_log_compaction.json records original/published hashes and
record counts. Original logs remain unchanged on the server.
