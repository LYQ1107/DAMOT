# ICS follow-up, 2026-10-29

This is a staged delivery. See `status.json` for completed and pending work. A pending experiment has no result.

Completed F analysis is in `../ics_probe/analysis_F_stage4_e5_v2/` and `../ics_probe/drift_decomposition/`. The original analysis remains intact. All numbers are measured; no expected outcome is substituted.

S2's covariance/PCS equivalence gate failed at every F checkpoint. At stage4 the largest principal angle is 1.751934 degrees, above the unchanged 1 degree threshold. The user approved reporting the actual stored `bank_pcs` separately from covariance eigenspace scans. Both families remain in the CSV and LaTeX tables.

The Fisher trace ratio uses unrenormalized projected vectors on car/test RoIs. Weather classifier inputs retain the earlier L2-normalization convention, and logistic probes are refitted after each intervention. Random k-dimensional results average ten bases. Weather-oracle rank is at most four.

S3 uses F stage1 fit videos as the common basis reference, literally as requested. No cross-model coordinate alignment is performed. Drift fractions are means of per-RoI energy fractions; absolute in/out energies and ratios of sums are exported separately. Cross-stage retrieval is a proxy, not measured tracking forgetting. The fixed figure shows stage3→4 old-domain energy and the equal mean of the three stage4←j retrieval cases; every other transition and dimension is retained in CSV.

Two protocol details are explicitly disclosed: the stored bank is not the exact eigenspace of the saved covariance; and HF saved exact APR histories in `stage3_final_rank*.pth`, despite lacking a `<stage>_roi_proto.pth` filename. S5 will use the same exact rank0 history file for all groups rather than approximate reconstruction. Its hash is in provenance.

Source data, features, and checkpoints remain on the server. No `.npz` or model weights are committed. `.npy` files contain the requested small projection bases only.

## Approved F tracking baseline

The historical A100 F metrics are not exactly reproduced by new V100 inference. Re-evaluating the historical prediction file with the current evaluator exactly reproduces its HOTA/DetA/AssA, while the newly generated predictions differ slightly. The precise cause is not fully isolated; the historical environment differs and its checkpoint binary hash was not recorded at evaluation time.

The user explicitly approved using this run's native FP32 beta0 as the F intervention reference and publishing every old/new difference. See `S4/F/fresh_vs_historical_baseline.csv`, the audit, preserved failed gate and authorization. New beta0 replay is checked against every native frame and against all eight car/OVERALL reference metrics per domain. HF retains its own frozen V100 reference. Nonzero result aggregation remains pending until every cell completes.

## Runtime source and short training

The causal head, bank and prototype-memory support files are unchanged copies of the already trained HF runtime. They are mirrored into the repository for checkpoint and APR-history compatibility. The short-stage entrypoint asserts resume stage4, two epochs, four GPUs, batch capacity16 and the original seed; it skips stages1–3. The historical full-episode helper supplies shared APIs; this follow-up does not invoke its full-training mode. Small saved optimizer-serialization and checkpoint-retention changes affect storage only.

Operational scripts assume the documented RUN/code and RUN/scripts layout and retain explicit read-only input paths. Source fingerprints and per-job launch records are retained on the server. Unrelated runtime modules are not copied into the repository.

S4 caches original FP32 detections and association embeddings once, then uses the unchanged native tracker for every projection. Projection applies to every native association embedding; bases are estimated from car RoIs. The enabled native head is independently compared against replay on the predeclared first Overcast video (ICS2 beta.5 and identity16 beta1). No normalization is added. Detector outputs are fixed; native track acceptance can still change tracking DetA.
