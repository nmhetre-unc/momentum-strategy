# SPY strategy comparison

Ticker: SPY · Window: 2008-01-01 to 2025-01-01 · Cost: 5bps one-way (buy_and_hold at 0bps) · n_boot=2000 · n_trials=5 · mean_block=20

| Strategy | Gross Sharpe | Net Sharpe | Net Sharpe 90% CI | Diff vs buy_and_hold (90% CI) | DSR vs 0 | DSR vs bench | Ann. Return | Ann. Vol | Net return | Max DD | Turnover | Trades | Skew | Kurtosis |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| buy_and_hold | 0.61 | 0.61 | [0.25, 1.01] | — | 0.903 | — | 10.7% | 20.0% | 458.2% | -51.9% | 0.1 | 1 | -0.07 | 17.2 |
| sma_crossover | 0.73 | 0.73 | [0.35, 1.17] | +0.12 [-0.16, +0.45] | 0.957 | 0.236 | 9.4% | 13.7% | 358.5% | -33.7% | 0.9 | 15 | -0.86 | 24.3 |
| momentum | 0.74 | 0.64 | [0.28, 1.00] | +0.03 [-0.32, +0.36] | 0.922 | 0.146 | 6.8% | 11.2% | 203.8% | -21.5% | 22.4 | 380 | -0.45 | 10.5 |
| mean_reversion | 0.48 | 0.46 | [0.13, 0.88] | -0.14 [-0.37, +0.11] | 0.764 | 0.038 | 6.5% | 16.5% | 192.1% | -47.7% | 5.2 | 88 | 0.20 | 34.8 |
| ml_direction | 0.72 | 0.70 | [0.36, 1.09] | +0.10 [-0.01, +0.20] | 0.955 | 0.214 | 12.2% | 18.9% | 608.9% | -46.6% | 6.8 | 115 | -0.05 | 17.7 |
| regime_filtered | 0.73 | 0.73 | [0.35, 1.17] | +0.12 [-0.16, +0.45] | 0.957 | 0.236 | 9.4% | 13.7% | 358.5% | -33.7% | 0.9 | 15 | -0.86 | 24.3 |
| regime_switch | 0.75 | 0.73 | [0.37, 1.19] | +0.13 [-0.20, +0.51] | 0.961 | 0.247 | 8.7% | 12.5% | 314.3% | -28.3% | 4.8 | 82 | -0.46 | 37.4 |
| regime_parameters | 0.81 | 0.79 | [0.44, 1.17] | +0.19 [-0.13, +0.50] | 0.978 | 0.329 | 9.2% | 12.0% | 342.7% | -24.7% | 3.4 | 57 | -0.63 | 8.9 |
| volatility_targeted | 0.84 | 0.83 | [0.47, 1.22] | +0.22 [-0.08, +0.54] | 0.985 | 0.388 | 8.9% | 11.0% | 326.7% | -18.7% | 2.1 | 922 | -0.71 | 7.0 |
| regime_sized | 0.78 | 0.78 | [0.40, 1.18] | +0.17 [-0.14, +0.49] | 0.973 | 0.305 | 8.1% | 10.7% | 273.2% | -22.5% | 1.8 | 113 | -0.81 | 12.0 |
| adaptive_ensemble | 0.96 | 0.93 | [0.59, 1.32] | +0.32 [-0.05, +0.69] | 0.995 | 0.549 | 7.2% | 7.8% | 225.6% | -13.6% | 5.3 | 910 | -0.22 | 14.8 |
| ml_regime_conditional | 0.72 | 0.63 | [0.28, 1.02] | +0.03 [-0.11, +0.16] | 0.920 | 0.138 | 10.7% | 18.9% | 462.2% | -46.6% | 34.5 | 585 | -0.00 | 20.2 |

## Mean block length sensitivity

Net Sharpe 90% CI at n_boot=2000, for mean_block in (10, 20, 40). A CI that moves a lot with mean_block means the width itself isn't trustworthy at any single block length.

| Strategy | mean_block=10 | mean_block=20 | mean_block=40 |
|---|---|---|---|
| sma_crossover | [0.33, 1.17] | [0.35, 1.17] | [0.34, 1.16] |
| adaptive_ensemble | [0.55, 1.33] | [0.59, 1.32] | [0.56, 1.30] |

## Passive-equivalent comparison: volatility_targeted

volatility_targeted's average exposure is 71.7%. Comparing it against a constant position held at that same exposure the whole period isolates the value of dynamic sizing from the trivial effect of just holding less.

| | Sharpe | Ann. Vol | Max DD |
|---|---|---|---|
| volatility_targeted (actual) | 0.832 | 11.0% | -18.7% |
| Passive @ 71.7% exposure | 0.607 | 14.3% | -39.7% |

## Rolling walk-forward: refit-per-fold vs fixed-model

train_days=756, test_days=126, cost_bps=5. Mean Sharpe across out-of-sample folds -- not the full-period Sharpe in the table above, and not affected by it either way. 'Fixed-model' reuses one fit across every fold (the old default); 'refit per fold' refits fresh on each fold's own training window, which is honest but n_folds times more expensive.

| Strategy | Fixed-model mean fold Sharpe | Refit-per-fold mean fold Sharpe | Change |
|---|---|---|---|
| regime_switch | 1.35 | 0.86 | -0.49 |
| adaptive_ensemble | 1.26 | 0.79 | -0.48 |
| regime_filtered | 1.08 | 0.66 | -0.42 |
| ml_regime_conditional | 1.19 | 0.99 | -0.20 |
| ml_direction | 1.37 | 1.18 | -0.19 |
| volatility_targeted | 0.97 | 0.97 | -0.00 |
| sma_crossover | 1.08 | 1.08 | +0.00 |
| momentum | 0.85 | 0.85 | +0.00 |
| mean_reversion | 1.35 | 1.35 | +0.00 |
| regime_sized | 0.95 | 0.99 | +0.04 |
| regime_parameters | 0.89 | 0.96 | +0.06 |
