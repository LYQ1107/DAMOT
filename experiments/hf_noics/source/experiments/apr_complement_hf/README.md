# F + contrastive noise + complement APR

This experiment starts from the existing Clear source epoch20, then adapts
Overcast -> Rainy -> Dawn-Dusk -> Night, five epochs/domain, on four GPUs.
No source training or seed sweep is performed. The run records the F seed.

The F student low-frequency pipeline and all original geometric transforms
are retained. The contrastive pipeline additionally applies the existing
`high_freq_noise` transform with noise_ratio=0.03 before Normalize. This
transform is spatial Gaussian noise mixing, not a Fourier high-pass filter.
The F ICS estimator, thresholds, prototype selection and loss weights remain.

APR takes an orthonormal basis E spanning the current detached ICS PCs.
The run uses beta=1 and the cosine loss between normalize((I-EE^T)f) and
normalize((I-EE^T)p), with nearest top-3 prototypes selected in that space.
The projection complement is a proxy for identity-preserving directions;
the existing class-centered ICS estimator does not prove weather separation.
The prototype pool remains history-only and car-only. A missing valid basis
skips APR instead of silently reverting to full-space alignment.

The supplied apr_coupling.patch remains archived with its SHA256. Its
projected_l2 objective on normalized features is also implemented and remains
the default for the three `...coupled...` reference configs. The new high-noise
experiment explicitly uses projected_cosine, which normalizes after projection
so beta=1 also removes the raw-feature gradient component in span(E).
Only this one configuration is trained; the reference configs are not a sweep.

The CPU unit test covers exact legacy beta=0 behavior, the reference distance
decomposition, complement invariance/gradient, QR after EMA, small-RoI use,
history-only collection, and candidate/history serialization:

    python tools/analysis/test_apr_coupling.py

The run is isolated under:

    /data1/liuyeqiang/damot_apr_complement_hf_20261028_1420

It reuses the read-only environment, source checkpoint, annotations and image
cache of `/data1/liuyeqiang/damot_rebuttal_20261026`. Missing cached videos
resolve to the original train/val NAS split through a new per-video view.
No original checkpoint, historical directory or annotation is edited.

Runtime: 4 V100 32GB, FP32, 16 actual pairs/GPU (nominal total batch64),
2 loader workers/GPU. Non-reentrant backbone activation recomputation fits
the original batch on V100; frozen BN statistics remain unchanged.
cuDNN benchmark is disabled. The restored software versions differ from
the original A100 training and are recorded. This is not a bit-exact replay.

Entrypoints (with the isolated run's code directory as cwd):

    python tools/run/apr_complement_experiment.py --run-root RUN_ROOT --mode prepare
    CUDA_VISIBLE_DEVICES=3,4,5,6 python -m torch.distributed.launch --nproc_per_node=4 --use_env tools/run/apr_complement_experiment.py --run-root RUN_ROOT --mode pilot --steps 3
    python tools/run/control_apr_complement.py --run-root RUN_ROOT --gpus 3,4,5,6

The pilot uses disposable F_stage3 weights and a documented reconstructed real
history to verify the active APR path. It never produces an accuracy model.
Formal training always initializes again from source S and an empty history.

The controller runs four-stage training, then ten learned-domain frozen
student evaluations. Thresholds are detector=.05, tracker-init=.30, obj=.20;
inference is native QDTrack with Conv-BN fusion and tracker reset per video.
HOTA indexes every frame, including empty-GT frames; plotting is disabled.
Results include full-precision class/OVERALL metrics, per-video predictions,
comparison_to_F.csv, stage_domain_metrics.csv and forgetting_metrics.csv.
No Clear holdout is used for adaptation, checkpoint selection or tuning.

Epoch recovery stores each rank's own APR history/candidates, ICS buffers and
main-process RNG. DataLoader worker RNG is not bit-exact after recovery.
The latest two model/optimizer epoch checkpoints per stage are retained to
fit local disk limits; final endpoints are hard links, not duplicate weights.
Controllers use a lock and inspect the prior child PID before launching.
Failed logs remain. Completion requires all ten cells and frame/video counts.

High noise and APR change together in this requested run. Its difference
versus archived F cannot be interpreted as an APR-only causal ablation.
