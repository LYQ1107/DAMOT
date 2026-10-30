# B3 nuisance release vs native ICS coupling

**Single training seed: 1790355234.** B3 and R each train Night for two epochs with four V100s, 16 samples/GPU and 2 workers/GPU. B0 reuses X3-0 without retraining.

Status: all requested experiments complete.

This follow-up was separately authorized after the original idAPR identity gate failed. B1/B2 and multi-seed experiments are outside this task.

B3 disables APR–ICS coupling and releases the fixed teacher-estimated nuisance-2 space. R retains native HF coupling (beta=1, basis=ICS, metric=projected_cosine), with space_mode=full. Both retain the original ICS settings and the same SHA256-identified three-domain APR history.

All values and differences are percentage points. F = HF stage3 old-domain value minus the final value; AvgF is the equal average of Overcast, Rainy and Dawn/Dusk, retaining negative values. The four-number tuples use the order **HOTA / DetA / IDF1 / AssA**.

| Group | Domain | HOTA | DetA | IDF1 | AssA | F (HOTA / DetA / IDF1 / AssA) | AvgF (same order) |
|---|---|---:|---:|---:|---:|---|---|
| B0 | night | 33.282830 | 28.681273 | 39.678051 | 39.275171 | — | 0.205479 / -0.067029 / 0.570163 / 0.310113 |
| B0 | overcast | 37.497778 | 30.149957 | 42.111432 | 46.980249 | 0.029906 / -0.262264 / 0.402240 / 0.490020 | — |
| B0 | rainy | 39.466803 | 31.827731 | 44.608083 | 50.646311 | 0.462235 / -0.094884 / 0.786862 / 0.360282 | — |
| B0 | dawn_dusk | 42.186656 | 37.970821 | 49.703742 | 47.382786 | 0.124296 / 0.156062 / 0.521386 / 0.080036 | — |
| R | night | 33.488683 | 28.550249 | 39.955170 | 39.964835 | — | 0.094544 / 0.082767 / 0.307094 / -0.059270 |
| R | overcast | 37.502287 | 29.994694 | 42.242182 | 47.233227 | 0.025397 / -0.107002 / 0.271489 / 0.237041 | — |
| R | rainy | 39.473064 | 31.605477 | 44.723459 | 50.672401 | 0.455973 / 0.127370 / 0.671486 / 0.334192 | — |
| R | dawn_dusk | 42.508690 | 37.898952 | 50.246820 | 48.211866 | -0.197739 / 0.227931 / -0.021692 / -0.749043 | — |
| B3 | night | 33.688937 | 28.800435 | 40.017019 | 40.086449 | — | 0.142899 / -0.106531 / 0.384967 / 0.263535 |
| B3 | overcast | 37.378345 | 30.136622 | 42.001309 | 46.712838 | 0.149338 / -0.248929 / 0.512362 / 0.757431 | — |
| B3 | rainy | 39.767349 | 31.963292 | 45.059388 | 51.071879 | 0.161689 / -0.230445 / 0.335557 / -0.065286 | — |
| B3 | dawn_dusk | 42.193283 | 37.967102 | 49.918147 | 47.364363 | 0.117668 / 0.159781 / 0.306981 / 0.098459 | — |

Complete per-class values and differences: [metrics.csv](metrics.csv), [mean_forgetting.csv](mean_forgetting.csv), [comparison.csv](comparison.csv).

Comparisons exceeding the 0.3 HOTA threshold (B3 minus reference; positive old-forgetting reduction means less forgetting): night vs B0: +0.406107 (B3 higher); rainy vs B0: +0.300546 (B3 higher); dawn_dusk vs R: -0.315407 (B3 lower). All remaining comparisons are “无差异”.
These are single-seed observations, not estimates of statistical significance.

Verification and basis provenance:

- U1: passed.
- U2: passed.
- extractor_parity: passed.
- warmstart: passed.
- checkpoint_serialization: passed.
- U3: passed, 50 updates, APR finite/nonzero in every iteration, history_domains=3. Mean weighted APR 0.001936145; same-step full-space reference 0.002077905; release/reference ratio 0.931777; mean in-space share 0.067750.
- Existing B0 log reference: first 40 available iterations, mean APR 0.002110000; B3/native-B0 ratio 0.917604. This reference is unpaired and uses four GPUs. The 50-step same-feature full-space reference above provides the paired scaling check.
- Teacher basis: 2851 RoIs, 870 images, 33 videos with proposals, score ≥0.75; all instance IDs are -1. The plan covers all 33 Dawn/Dusk videos, 990 sampled frames with the original extractor frame sampling.
- Basis SHA256: `9aebdbe0790d5ee4a8f1504cbdcd53b966699666fe601e496c4c7eb9ad34a7ba`. Principal angles to A/HF-stage3 nuisance-2: 57.923655, 76.756487 degrees. A used FAP-only GT-box features on fit videos across domains; this training basis uses teacher boxes and three perturbation views on Dawn/Dusk, as newly requested.

Protocol notes and deviations:

- U3 does not retrain B0. Its archived log has 20-step averages, so the native comparison uses the available first40 steps; an exact 50-step same-feature full-space counterfactual is additionally recorded. Smoke weights are discarded and are never formal starting weights.
- Default full losses were checked bitwise on the same serialized train batch; GT extraction was checked on two fixed frames, and teacher extraction was replayed with all GT boxes removed.
- The actual X3 HF runtime is copied under this task. The original HF finite-loss, EMA and ICS audit hooks are retained; the extra audit only records losses and fixed-basis state.
- Independent read-only SSHFS under this task replaces a stale former private mount. The current shared mount and every previous task directory remain unchanged.
- Native frozen inference, fusion, thresholds .05/.30/.20 and all-frame HOTA are unchanged. HF stage3 Overcast/Rainy metrics exactly match all 56 archived class/metric reference values; Dawn/Dusk baseline and B0 Night/Dawn evaluations are reused as requested.
- After each group finishes all four evaluations, only its own redundant epoch1 and stage00 source copies are removed. Epoch2, the final endpoint and stage3 starting copy are retained.
- Final file validation detected that this old MMCV deliberately serializes nonpersistent buffers. The original B3 checkpoint contains two fixed basis buffers, despite the original audit flag (which checked PyTorch state_dict only). The final code adds a release-only state-dict hook and checks MMCV directly. The actual training versions of the two affected source files are preserved in source_training/, with their original pre-training seal and code_training.patch. No forward, loss or update code changed. The clean B3 epoch_2_export.pth omits only these two buffers; all 752 retained tensors and all other checkpoint payload are bitwise identical. Evaluations reference the preserved original checkpoints and remain valid; no training or inference was repeated for this serialization fix.

See `inputs.txt`, `runs/*/config_diff.json`, `U/`, `basis/`, `training_audit.csv` and `manifests/` for exact source paths and checksums. Model weights, raw predictions and feature NPZ files remain on the server.
