# Historical ICS comparison after authorized inference completion

Twelve old-domain cells were inferred from the two existing no-ICS runs. No historical training was repeated. Original native four-GPU evaluation, Conv-BN fusion, cuDNN benchmark=True, tracker init .30 and the original legacy HOTA frame mapping are retained. New missing inference uses V100s; the archived runs/evaluations used A100s. Historical seed and APR differences remain confounders.

All values are percentage points. AIP = equal mean of stage-wise seen-domain means. AvgF = mean over three old domains of max(R_sj, s=j..3) minus R_4j, without clipping. FinalOld = stage4 mean over three old domains. BWT = mean(R_4j-R_jj). DiagonalMean = mean(R_jj), distinct from stage4 FinalMean.

| Run | Metric | Seed | DiagonalMean | FinalMean | AIP | AvgF | FinalOld | BWT | ΔFinalMean vs no ICS | ΔAvgF vs no ICS |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| SPA_pipeline | HOTA | 1929377231 | 37.375345 | 37.901504 | 37.989911 | 0.329849 | 39.547482 | 0.701546 | — | — |
| SPA_pipeline | DetA | 1929377231 | 31.618715 | 32.447272 | 31.068988 | -0.284349 | 33.554971 | 1.104742 | — | — |
| SPA_pipeline | IDF1 | 1929377231 | 42.665395 | 43.704976 | 43.154926 | 0.331166 | 45.537119 | 1.386108 | — | — |
| SPA_pipeline | AssA | 1929377231 | 44.916133 | 45.095617 | 47.102183 | 1.049719 | 47.422360 | 0.239312 | — | — |
| SPA_legacy | HOTA | 1468082502 | 37.146465 | N/A | N/A | N/A | N/A | N/A | — | — |
| SPA_legacy | DetA | 1468082502 | 31.720609 | N/A | N/A | N/A | N/A | N/A | — | — |
| SPA_legacy | IDF1 | 1468082502 | 42.128925 | N/A | N/A | N/A | N/A | N/A | — | — |
| SPA_legacy | AssA | 1468082502 | 44.386453 | N/A | N/A | N/A | N/A | N/A | — | — |
| SPA_oldICS | HOTA | 1591423161 | 38.123603 | 38.263927 | 38.363765 | 0.511265 | 39.607086 | 0.187098 | +0.362423 | +0.181415 |
| SPA_oldICS | DetA | 1591423161 | 31.711049 | 32.242963 | 30.982086 | -0.142865 | 33.258704 | 0.709219 | -0.204309 | +0.141484 |
| SPA_oldICS | IDF1 | 1591423161 | 44.120230 | 44.370870 | 43.758983 | 0.455794 | 45.552279 | 0.334186 | +0.665893 | +0.124628 |
| SPA_oldICS | AssA | 1591423161 | 46.489761 | 46.116275 | 48.140106 | 1.899028 | 47.838853 | -0.497981 | +1.020658 | +0.849308 |
| SPA_APR | HOTA | 1731334941 | 38.101083 | 38.314976 | 38.229880 | 0.254937 | 39.576630 | 0.285190 | — | — |
| SPA_APR | DetA | 1731334941 | 31.567656 | 32.424271 | 30.872624 | -0.464423 | 33.439824 | 1.142154 | — | — |
| SPA_APR | IDF1 | 1731334941 | 44.135496 | 44.587890 | 43.608444 | 0.127961 | 45.630739 | 0.603192 | — | — |
| SPA_APR | AssA | 1731334941 | 46.639038 | 45.951680 | 47.988360 | 1.581933 | 47.530995 | -0.916477 | — | — |
| F_old | HOTA | 952475529 | 38.004812 | 38.246856 | 38.230228 | 0.173532 | 39.692017 | 0.322725 | -0.068120 | -0.081405 |
| F_old | DetA | 952475529 | 31.536425 | 32.235887 | 30.919055 | -0.448536 | 33.391503 | 0.932615 | -0.188385 | +0.015887 |
| F_old | IDF1 | 952475529 | 43.827019 | 44.035502 | 43.438658 | 0.393332 | 45.394537 | 0.277978 | -0.552388 | +0.265371 |
| F_old | AssA | 952475529 | 46.505990 | 46.370306 | 48.032740 | 0.974717 | 48.298250 | -0.180911 | +0.418626 | -0.607215 |
| causalShrinkage | HOTA | 1776369864 | 38.227751 | 38.227364 | 38.380850 | 0.344418 | 39.562458 | -0.000516 | — | — |
| causalShrinkage | DetA | 1776369864 | 31.556221 | 32.626486 | 30.943399 | -0.921453 | 33.748246 | 1.427020 | — | — |
| causalShrinkage | IDF1 | 1776369864 | 44.141492 | 44.221913 | 43.755202 | 0.516201 | 45.451277 | 0.107227 | — | — |
| causalShrinkage | AssA | 1776369864 | 47.025007 | 45.428152 | 48.227911 | 2.550779 | 47.005906 | -2.129139 | — | — |
| causalProtoCenter | HOTA | 388086182 | 37.998570 | 37.765906 | 38.136077 | 0.803352 | 39.008964 | -0.310219 | — | — |
| causalProtoCenter | DetA | 388086182 | 31.501347 | 31.750150 | 30.853240 | 0.173686 | 32.803749 | 0.331737 | — | — |
| causalProtoCenter | IDF1 | 388086182 | 43.909030 | 43.679836 | 43.447606 | 0.901121 | 44.733839 | -0.305592 | — | — |
| causalProtoCenter | AssA | 388086182 | 46.461378 | 45.546733 | 47.744693 | 2.048272 | 47.020864 | -1.219527 | — | — |
| causalProtoShrink | HOTA | 1790355234 | 38.128815 | 38.344555 | 38.263235 | 0.171526 | 39.695761 | 0.287653 | — | — |
| causalProtoShrink | DetA | 1790355234 | 31.371061 | 31.900624 | 30.764481 | -0.287272 | 33.071458 | 0.706084 | — | — |
| causalProtoShrink | IDF1 | 1790355234 | 44.180157 | 44.332798 | 43.612511 | 0.232338 | 45.516186 | 0.203522 | — | — |
| causalProtoShrink | AssA | 1790355234 | 46.978884 | 46.799338 | 48.161960 | 0.937174 | 48.391345 | -0.239394 | — | — |

Δ is old ICS minus its corresponding no-ICS run. Lower AvgF is better. Both comparisons use different historical seeds. F_old also has weak APR (.005, score .75, 64 RoIs, six GPUs ×14), versus SPA_APR (.01, score .70, 128 RoIs, four GPUs ×16). The APR implementations also differ: SPA_APR uses ProtoQuasiDenseTrackHead/RoIPrototypeMemory; F_old uses CausalQuasiDenseTrackHead/StageRoIPrototypeMemory. These are descriptive comparisons; C0–C4 use a common seed and no APR.

SPA_legacy was not requested for supplemental inference and keeps its existing diagonal results; missing cross-domain fields remain N/A. Future-domain cells were not added. All archived diagonal values are retained unchanged. These legacy results are reported separately from the all-frame HOTA used for the new C-series/X3 evaluations.

Seed audit: ProtoCenter stage1 also has an earlier metadata-only startup log with seed884315508 and no logged train iterations. The epoch5-completed log and resolved configuration use 388086182, which is the seed reported in the table. Both records are retained; auxiliary startup seeds are separated from completed-run seeds in resolved_config_audit.json.
