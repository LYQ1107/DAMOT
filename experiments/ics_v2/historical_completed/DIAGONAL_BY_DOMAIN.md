# Per-domain diagonal scores

Each cell is HOTA / AssA, from the original legacy evaluator at the end of its own adaptation stage. R11=Overcast, R22=Rainy, R33=Dawn/Dusk, R44=Night. Historical seeds differ.

| Run | Seed | R11 Overcast | R22 Rainy | R33 Dawn/Dusk | R44 Night |
|---|---:|---:|---:|---:|---:|
| SPA/FAP，无 ICS、无 APR | 1929377231 | 36.058875 / 46.880755 | 38.564129 / 47.965632 | 41.914807 / 46.702756 | 32.963568 / 38.115388 |
| 旧版 SPA，无 ICS、无 APR | 1468082502 | 37.136919 / 47.266209 | 38.817486 / 48.971414 | 41.548872 / 46.274505 | 31.082584 / 35.033685 |
| SPA/FAP + 老 ICS | 1591423161 | 36.535794 / 47.829834 | 39.427438 / 49.508267 | 42.296730 / 47.672400 | 34.234451 / 40.948544 |
| SPA/FAP + APR，无 ICS | 1731334941 | 36.238939 / 47.766159 | 39.475755 / 50.041033 | 42.159626 / 47.535222 | 34.530012 / 41.213736 |
| F_old（老 ICS + APR） | 952475529 | 36.769113 / 47.554501 | 39.057944 / 50.127799 | 42.280817 / 47.755184 | 33.911372 / 40.586475 |
| causalShrinkage + APR | 1776369864 | 36.793155 / 48.532144 | 39.676776 / 51.011717 | 42.218991 / 47.861275 | 34.222083 / 40.694891 |
| causalProtoCenter + APR | 388086182 | 36.620279 / 47.709356 | 39.198678 / 49.385794 | 42.138590 / 47.626022 | 34.036731 / 41.124340 |
| causalProtoShrink + APR | 1790355234 | 36.434644 / 47.442346 | 39.458509 / 50.224196 | 42.331171 / 48.225675 | 34.290935 / 42.023319 |

Full precision: diagonal_by_domain.csv. All means match the previously reported diagonal means. The new C-series separately uses the X3 all-frame evaluator.
