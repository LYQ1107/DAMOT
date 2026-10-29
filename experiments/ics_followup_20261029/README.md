# Completed ICS follow-up

All required primary S1–S5 experiments are complete. The five result tables and input/checkpoint/seed/source provenance are in `../SUMMARY_20261029.md`; full tables are in `tables/`. Original v1 results remain intact.

The user approved two protocol resolutions: report stored bank_pcs separately from covariance eigenspaces when the original <1 degree gate fails; and use fresh native V100 beta0 for F tracking interventions while disclosing every old/new difference. Thresholds and seeds were not tuned. The failed checks, diagnostics and decisions are retained.

S1 configurations contain effective model pipelines and data.train pipelines. NOGEOM has no active stochastic geometry. Offline probes fix geometry and use exactly the same RoIs as F. S2 uses raw projected car/test vectors for Fisher, a complete multi-rank scan, ten random bases for P3, all P2 binary spaces and the Full confusion matrix. P1 retains its original fifty-basis random control. BDD domain offsets are unpaired and include scene/identity differences.

S3 uses the literal common F stage1 fit reference and test statistics; drift shares are per-RoI fraction means over nonzero drifts, with absolute in/out energies also exported. Every transition and restoration dimension is retained. Retrieval forgetting is a proxy; tracking forgetting is measured separately in S5.

S4 replays the unchanged native tracker using original FP32 embeddings/detections. Native beta0 output equality and reference metrics are checked before nonzero beta. Independent enabled-head checks cover the predeclared first Overcast video at ICS2 beta.5 and identity16 beta1. Projection applies to all association embeddings, with no added normalization; bases are estimated from car RoIs. FAP2 uses clear fit videos only. Detector outputs stay fixed, while native track acceptance can change DetA.

S5 executes only Night for two epochs, from the same complete HF stage3 checkpoint, same historical APR prototypes and seed. Inherited banks are checked before the first update; every insufficient-pairs event is counted. Storage changes retain the final epoch and omit optimizer serialization. Exact historical APR vectors were available in rank-local runtime sidecars despite the absence of a <stage>_roi_proto.pth filename, so no approximate reconstruction is used. The optional NOGEOM short-training repeat was not run.

Small .npy files contain projection bases. Raw .npz features, model weights, large prediction caches and full local source archives remain on the server. Per-job fingerprints, configurations, metadata, logs, CSV and LaTeX are delivered. Operational scripts use the documented RUN/code layout and explicit read-only input roots.
