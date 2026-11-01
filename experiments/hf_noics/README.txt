Single-seed HF ablation: disable ICS and APR coupling, keep FAP + original APR.
Seed1790355234; Clear S -> Overcast -> Rainy -> Dawn/Dusk -> Night, five epochs/domain.
The local code/ snapshot (published as source/) is byte-identical to the actual HF source, with442 files /346 Python files.
Only two model config fields change: use_causal_direction_bank=False and roi_proto_couple_ics=False.
External scripts adapt None-safe telemetry and two capability assertions, preserving all numerical training operations.
The fixed simplification decision is at the start of SUMMARY.md; it uses OVERALL final four-domain mean HOTA and HOTA AvgF.
All comparisons and video confidence intervals use one network training seed. Bootstrap/split seed0 is for offline statistics only.
Frozen evaluation protocol and continual-learning definitions match experiments/hf_intics/.
Model weights, predictions and input images remain on the experiment storage; reviewable source/config/audits/CSVs go to experiments/hf_noics/.
