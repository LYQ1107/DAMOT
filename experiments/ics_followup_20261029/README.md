# ICS follow-up, 2026-10-29

This is a staged delivery. See `status.json` for completed and pending work. A pending experiment has no result.

Completed F analysis is in `../ics_probe/analysis_F_stage4_e5_v2/` and `../ics_probe/drift_decomposition/`. The original analysis remains intact. All numbers are measured; no expected outcome is substituted.

S2's covariance/PCS equivalence gate failed at every F checkpoint. At stage4 the largest principal angle is 1.751934 degrees, above the unchanged 1 degree threshold. The user approved reporting the actual stored `bank_pcs` separately from covariance eigenspace scans. Both families remain in the CSV and LaTeX tables.

The Fisher trace ratio uses unrenormalized projected vectors on car/test RoIs. Weather classifier inputs retain the earlier L2-normalization convention, and logistic probes are refitted after each intervention. Random k-dimensional results average ten bases. Weather-oracle rank is at most four.

S3 uses F stage1 fit videos as the common basis reference, literally as requested. No cross-model coordinate alignment is performed. Drift fractions are means of per-RoI energy fractions; absolute in/out energies and ratios of sums are exported separately. Cross-stage retrieval is a proxy, not measured tracking forgetting. The fixed figure shows stage3→4 old-domain energy and the equal mean of the three stage4←j retrieval cases; every other transition and dimension is retained in CSV.

Two protocol details are explicitly disclosed: the stored bank is not the exact eigenspace of the saved covariance; and HF saved exact APR histories in `stage3_final_rank*.pth`, despite lacking a `<stage>_roi_proto.pth` filename. S5 will use the same exact rank0 history file for all groups rather than approximate reconstruction. Its hash is in provenance.

Source data, features, and checkpoints remain on the server. No `.npz` or model weights are committed. `.npy` files contain the requested small projection bases only.
