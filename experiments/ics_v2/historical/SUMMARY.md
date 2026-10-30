# Existing four-domain five-epoch runs (no training or inference)

Uniform archived legacy evaluator; original full-precision CSVs are used when available (per-cell provenance records precision). Do not mix these values with the all-frame HOTA of the new X3-protocol runs.

AIP = mean of four seen-domain stage means. FinalMean = stage4 four-domain mean. FinalOld = stage4 three old-domain mean. AvgF = mean(best prior learned-stage value, excluding stage4, minus stage4), retaining negative values. BWT = mean(stage4 minus diagonal) over three old domains. DiagonalMean is diagnostic; it is never substituted for FinalMean.

| Run | Metric | Seed | Cells | FinalMean | AIP | AvgF | FinalOld | BWT |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| SPA_pipeline | HOTA | 1929377231 | 4/16 | N/A | N/A | N/A | N/A | N/A |
| SPA_pipeline | DetA | 1929377231 | 4/16 | N/A | N/A | N/A | N/A | N/A |
| SPA_pipeline | IDF1 | 1929377231 | 4/16 | N/A | N/A | N/A | N/A | N/A |
| SPA_pipeline | AssA | 1929377231 | 4/16 | N/A | N/A | N/A | N/A | N/A |
| SPA_legacy | HOTA | 1468082502 | 4/16 | N/A | N/A | N/A | N/A | N/A |
| SPA_legacy | DetA | 1468082502 | 4/16 | N/A | N/A | N/A | N/A | N/A |
| SPA_legacy | IDF1 | 1468082502 | 4/16 | N/A | N/A | N/A | N/A | N/A |
| SPA_legacy | AssA | 1468082502 | 4/16 | N/A | N/A | N/A | N/A | N/A |
| SPA_oldICS | HOTA | 1591423161 | 16/16 | 38.263927 | 38.363765 | 0.511265 | 39.607086 | 0.187098 |
| SPA_oldICS | DetA | 1591423161 | 16/16 | 32.242963 | 30.982086 | -0.142865 | 33.258704 | 0.709219 |
| SPA_oldICS | IDF1 | 1591423161 | 16/16 | 44.370870 | 43.758983 | 0.455794 | 45.552279 | 0.334186 |
| SPA_oldICS | AssA | 1591423161 | 16/16 | 46.116275 | 48.140106 | 1.899028 | 47.838853 | -0.497981 |
| SPA_APR | HOTA | 1731334941 | 4/16 | N/A | N/A | N/A | N/A | N/A |
| SPA_APR | DetA | 1731334941 | 4/16 | N/A | N/A | N/A | N/A | N/A |
| SPA_APR | IDF1 | 1731334941 | 4/16 | N/A | N/A | N/A | N/A | N/A |
| SPA_APR | AssA | 1731334941 | 4/16 | N/A | N/A | N/A | N/A | N/A |
| F_old | HOTA | 952475529 | 16/16 | 38.246856 | 38.230228 | 0.173532 | 39.692017 | 0.322725 |
| F_old | DetA | 952475529 | 16/16 | 32.235887 | 30.919055 | -0.448536 | 33.391503 | 0.932615 |
| F_old | IDF1 | 952475529 | 16/16 | 44.035502 | 43.438658 | 0.393332 | 45.394537 | 0.277978 |
| F_old | AssA | 952475529 | 16/16 | 46.370306 | 48.032740 | 0.974717 | 48.298250 | -0.180911 |
| causalShrinkage | HOTA | 1776369864 | 16/16 | 38.227364 | 38.380850 | 0.344418 | 39.562458 | -0.000516 |
| causalShrinkage | DetA | 1776369864 | 16/16 | 32.626486 | 30.943399 | -0.921453 | 33.748246 | 1.427020 |
| causalShrinkage | IDF1 | 1776369864 | 16/16 | 44.221913 | 43.755202 | 0.516201 | 45.451277 | 0.107227 |
| causalShrinkage | AssA | 1776369864 | 16/16 | 45.428152 | 48.227911 | 2.550779 | 47.005906 | -2.129139 |
| causalProtoCenter | HOTA | 388086182 | 16/16 | 37.765906 | 38.136077 | 0.803352 | 39.008964 | -0.310219 |
| causalProtoCenter | DetA | 388086182 | 16/16 | 31.750150 | 30.853240 | 0.173686 | 32.803749 | 0.331737 |
| causalProtoCenter | IDF1 | 388086182 | 16/16 | 43.679836 | 43.447606 | 0.901121 | 44.733839 | -0.305592 |
| causalProtoCenter | AssA | 388086182 | 16/16 | 45.546733 | 47.744693 | 2.048272 | 47.020864 | -1.219527 |
| causalProtoShrink | HOTA | 1790355234 | 16/16 | 38.344555 | 38.263235 | 0.171526 | 39.695761 | 0.287653 |
| causalProtoShrink | DetA | 1790355234 | 16/16 | 31.900624 | 30.764481 | -0.287272 | 33.071458 | 0.706084 |
| causalProtoShrink | IDF1 | 1790355234 | 16/16 | 44.332798 | 43.612511 | 0.232338 | 45.516186 | 0.203522 |
| causalProtoShrink | AssA | 1790355234 | 16/16 | 46.799338 | 48.161960 | 0.937174 | 48.391345 | -0.239394 |

**Old ICS minus no ICS: FinalMean and AvgF differences are unavailable.** The no-ICS runs have four diagonal evaluations and no stage4 old-domain results. No missing run or evaluation was added. Available and missing cells are listed separately in the CSVs.

- SPA_oldICS vs SPA_pipeline: seeds 1591423161 vs 1929377231; same seed = False.
- F_old vs SPA_APR: seeds 952475529 vs 1731334941; same seed = False.

SPA_pipeline is the pipeline FAP baseline. SPA_legacy uses a different augmentation implementation and is separate. SPA_APR uses roiV1; F_old uses roiV1Weak, so APR is also not a controlled match. Every stage resolved config, causal/ROI fields, batch/GPU count, seed and epoch5 train log is retained with SHA256 provenance. Training completion and evaluation coverage are separate.
