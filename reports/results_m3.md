# Milestone 3 results: walk-forward evaluation

Pre-registered in DESIGN.md §16. Target `y` = YoY log revenue growth (first-filed); feature `x` = YoY log pageviews over the exact fiscal period. Skill = 1 − MAE/MAE(naive). Hit rate = acceleration direction; p one-sided vs 50%. DM < 0 means the model beats naive. All models are scored on the identical sample within each spec.

## Verdict
- **Level forecasts:** 0 of 16 regression specs beat the naive forecast with positive skill and DM p < 0.05. Best skill: -4%.
- **Direction:** the best edge over the base-rate benchmark (MB) is +1.5% (M3, m30/raw/all). Raw hit rates above 50% mostly reflect that decelerations outnumber accelerations.
- **Cross-sectional IC:** 0 of 8 specs have |t| > 2. Mean IC ranges 0.04 to 0.13.

## All specs
| cut | variant | sample | model | n | mae | skill | hit_rate | hit_p | hit_vs_base | dm_stat | dm_p |
|---|---|---|---|---|---|---|---|---|---|---|---|
| full | raw | all | M0 | 281 | 0.077 | 0% | – | – | – | – | – |
| full | raw | all | M1 | 281 | 0.112 | -44% | 54% | 0.116 | -4% | 1.873 | 0.061 |
| full | raw | all | M2 | 281 | 0.092 | -19% | 52% | 0.317 | -6% | 1.962 | 0.050 |
| full | raw | all | M3 | 281 | – | – | 57% | 0.008 | -0% | – | – |
| full | raw | all | MB | 281 | – | – | 58% | 0.006 | – | – | – |
| full | raw | ex_covid | M0 | 225 | 0.048 | 0% | – | – | – | – | – |
| full | raw | ex_covid | M1 | 225 | 0.057 | -18% | 57% | 0.016 | -4% | 2.346 | 0.019 |
| full | raw | ex_covid | M2 | 225 | 0.051 | -5% | 54% | 0.143 | -8% | 0.240 | 0.810 |
| full | raw | ex_covid | M3 | 225 | – | – | 61% | 0.001 | -0% | – | – |
| full | raw | ex_covid | MB | 225 | – | – | 61% | 0.000 | – | – | – |
| full | site | all | M0 | 281 | 0.077 | 0% | – | – | – | – | – |
| full | site | all | M1 | 281 | 0.109 | -41% | 54% | 0.076 | -3% | 1.803 | 0.071 |
| full | site | all | M2 | 281 | 0.093 | -19% | 50% | 0.500 | -7% | 2.049 | 0.040 |
| full | site | all | M3 | 281 | – | – | 57% | 0.012 | -1% | – | – |
| full | site | all | MB | 281 | – | – | 58% | 0.006 | – | – | – |
| full | site | ex_covid | M0 | 225 | 0.048 | 0% | – | – | – | – | – |
| full | site | ex_covid | M1 | 225 | 0.055 | -15% | 58% | 0.008 | -3% | 1.827 | 0.068 |
| full | site | ex_covid | M2 | 225 | 0.051 | -5% | 55% | 0.091 | -7% | 0.251 | 0.802 |
| full | site | ex_covid | M3 | 225 | – | – | 60% | 0.001 | -1% | – | – |
| full | site | ex_covid | MB | 225 | – | – | 61% | 0.000 | – | – | – |
| m30 | raw | all | M0 | 275 | 0.076 | 0% | – | – | – | – | – |
| m30 | raw | all | M1 | 275 | 0.110 | -44% | 55% | 0.046 | 0% | 1.758 | 0.079 |
| m30 | raw | all | M2 | 275 | 0.093 | -22% | 50% | 0.548 | -5% | 2.444 | 0.015 |
| m30 | raw | all | M3 | 275 | – | – | 56% | 0.020 | 1% | – | – |
| m30 | raw | all | MB | 275 | – | – | 55% | 0.058 | – | – | – |
| m30 | raw | ex_covid | M0 | 220 | 0.046 | 0% | – | – | – | – | – |
| m30 | raw | ex_covid | M1 | 220 | 0.055 | -18% | 59% | 0.006 | -2% | 2.110 | 0.035 |
| m30 | raw | ex_covid | M2 | 220 | 0.049 | -4% | 55% | 0.078 | -6% | 0.236 | 0.813 |
| m30 | raw | ex_covid | M3 | 220 | – | – | 60% | 0.002 | -1% | – | – |
| m30 | raw | ex_covid | MB | 220 | – | – | 61% | 0.001 | – | – | – |
| m30 | site | all | M0 | 275 | 0.076 | 0% | – | – | – | – | – |
| m30 | site | all | M1 | 275 | 0.109 | -43% | 54% | 0.092 | -1% | 1.774 | 0.076 |
| m30 | site | all | M2 | 275 | 0.094 | -23% | 52% | 0.315 | -3% | 2.625 | 0.009 |
| m30 | site | all | M3 | 275 | – | – | 56% | 0.020 | 1% | – | – |
| m30 | site | all | MB | 275 | – | – | 55% | 0.058 | – | – | – |
| m30 | site | ex_covid | M0 | 220 | 0.046 | 0% | – | – | – | – | – |
| m30 | site | ex_covid | M1 | 220 | 0.054 | -16% | 60% | 0.003 | -1% | 1.849 | 0.065 |
| m30 | site | ex_covid | M2 | 220 | 0.049 | -5% | 56% | 0.046 | -5% | 0.260 | 0.795 |
| m30 | site | ex_covid | M3 | 220 | – | – | 60% | 0.001 | -0% | – | – |
| m30 | site | ex_covid | MB | 220 | – | – | 61% | 0.001 | – | – | – |

## Cross-sectional IC of Δx vs Δy (per calendar quarter, ≥5 names)
| cut | variant | sample | mean_ic | ic_t | periods |
|---|---|---|---|---|---|
| full | raw | all | 0.059 | 0.798 | 29 |
| full | raw | ex_covid | 0.064 | 0.751 | 24 |
| full | site | all | 0.039 | 0.528 | 29 |
| full | site | ex_covid | 0.045 | 0.528 | 24 |
| m30 | raw | all | 0.119 | 1.516 | 28 |
| m30 | raw | ex_covid | 0.114 | 1.247 | 23 |
| m30 | site | all | 0.126 | 1.653 | 28 |
| m30 | site | ex_covid | 0.126 | 1.341 | 23 |

## Per ticker (primary spec: full cut, raw, all quarters)
| ticker | n | skill_M1 | skill_M2 | hit_rate_M1 | hit_rate_M2 | hit_rate_M3 | hit_rate_MB |
|---|---|---|---|---|---|---|---|
| CMG | 22 | -20% | -16% | 59% | 45% | 55% | 55% |
| ETSY | 29 | -69% | -50% | 55% | 41% | 52% | 52% |
| LULU | 26 | -10% | -0% | 62% | 62% | 50% | 54% |
| NFLX | 29 | -11% | -123% | 48% | 38% | 66% | 59% |
| PINS | 18 | -158% | -165% | 44% | 61% | 56% | 56% |
| PLNT | 29 | -58% | 6% | 48% | 62% | 55% | 62% |
| PTON | 16 | -42% | -14% | 62% | 69% | 50% | 44% |
| RBLX | 10 | -12% | 43% | 50% | 100% | 80% | 70% |
| SHAK | 30 | -35% | 5% | 53% | 50% | 57% | 57% |
| SNAP | 26 | -28% | -62% | 46% | 35% | 54% | 62% |
| WING | 29 | -10% | 10% | 55% | 59% | 62% | 62% |
| ZM | 17 | -425% | -530% | 65% | 29% | 65% | 65% |

## Leave one ticker out (primary spec)
| dropped | m2_skill | m2_hit | m3_hit |
|---|---|---|---|
| CMG | -19% | 53% | 56% |
| ETSY | -16% | 52% | 58% |
| LULU | -23% | 51% | 59% |
| NFLX | -17% | 53% | 59% |
| PINS | -16% | 51% | 57% |
| PLNT | -19% | 51% | 58% |
| PTON | -21% | 51% | 56% |
| RBLX | -22% | 50% | 57% |
| SHAK | -21% | 52% | 57% |
| SNAP | -24% | 53% | 57% |
| WING | -23% | 51% | 56% |
| ZM | -8% | 55% | 55% |

## In-sample coefficients (primary spec; HAC per ticker, clustered pooled)
| ticker | n | b_ylag | c_x | t_x | r2 |
|---|---|---|---|---|---|
| CMG | 34 | 0.533 | 0.006 | 0.365 | 0.295 |
| ETSY | 41 | 0.694 | 0.155 | 1.591 | 0.712 |
| LULU | 41 | 0.667 | -0.060 | -0.889 | 0.396 |
| NFLX | 41 | 0.989 | -0.026 | -2.337 | 0.902 |
| PINS | 30 | 0.732 | -0.028 | -0.425 | 0.523 |
| PLNT | 41 | 0.341 | 0.346 | 2.183 | 0.237 |
| PTON | 28 | 0.695 | 0.185 | 2.221 | 0.854 |
| RBLX | 22 | 0.732 | -0.014 | -0.175 | 0.720 |
| SHAK | 42 | 0.533 | 0.021 | 0.181 | 0.306 |
| SNAP | 38 | 0.661 | -0.026 | -0.834 | 0.744 |
| WING | 41 | 0.709 | 0.021 | 0.395 | 0.482 |
| ZM | 29 | 0.655 | 0.194 | 12.503 | 0.978 |
| POOLED (FE, clustered) | 428 | 0.661 | 0.117 | 2.561 | 0.649 |
