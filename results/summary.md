# SPY strategy comparison

Ticker: SPY · Window: 2008-01-01 to 2025-01-01 · Cost: 5bps one-way (buy_and_hold at 0bps) · n_boot=2000 · n_trials=5 · mean_block=20

## Full period: strategies that learn nothing

Rules with fixed parameters, scored on every day.

| Strategy | Gross Sharpe | Net Sharpe | Net Sharpe 90% CI | Diff vs buy_and_hold (90% CI) | DSR vs 0 | DSR vs bench | Ann. Return | Ann. Vol | Net return | Max DD | Turnover | Trades | Skew | Kurtosis |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| buy_and_hold | 0.61 | 0.61 | [0.25, 1.01] | — | 0.903 | — | 10.7% | 20.0% | 458.2% | -51.9% | 0.1 | 1 | -0.07 | 17.2 |
| sma_crossover | 0.73 | 0.73 | [0.35, 1.17] | +0.12 [-0.16, +0.45] | 0.957 | 0.236 | 9.4% | 13.7% | 358.5% | -33.7% | 0.9 | 15 | -0.86 | 24.3 |
| momentum | 0.74 | 0.64 | [0.28, 1.00] | +0.03 [-0.32, +0.36] | 0.922 | 0.146 | 6.8% | 11.2% | 203.8% | -21.5% | 22.4 | 380 | -0.45 | 10.5 |
| mean_reversion | 0.48 | 0.46 | [0.13, 0.88] | -0.14 [-0.37, +0.11] | 0.764 | 0.038 | 6.5% | 16.5% | 192.1% | -47.7% | 5.2 | 88 | 0.20 | 34.8 |
| volatility_targeted | 0.84 | 0.83 | [0.47, 1.22] | +0.22 [-0.08, +0.54] | 0.985 | 0.388 | 8.9% | 11.0% | 326.7% | -18.7% | 2.1 | 922 | -0.71 | 7.0 |

## Out of sample only: strategies that learn

Each strategy scored only from its first out-of-sample day, the day after everything it learned from (a fitted model, a regime model's fit window, a choice learned on past performance; see results/ml/learning_strategies_audit.json). buy_and_hold is measured over the identical window, and the difference and DSR vs bench are against it. Same n_trials=5. The first day's return is zero by construction (a position decided that day earns from the next).

| Strategy | Out of sample from | Days | Net Sharpe | Net Sharpe 90% CI | buy_and_hold Sharpe, same window | buy_and_hold 90% CI | Diff vs buy_and_hold (90% CI) | DSR vs 0 | DSR vs bench | Ann. Return | buy_and_hold Ann. Return | Max DD | buy_and_hold Max DD | Exposure | Trades |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| ml_direction | 2019-11-22 | 1284 | 0.84 | [0.25, 1.53] | 0.78 | [0.15, 1.60] | +0.06 [-0.22, +0.30] | 0.748 | 0.144 | 15.2% | 15.1% | -26.2% | -33.7% | 98.1% | 39 |
| regime_filtered | 2020-04-08 | 1191 | 1.33 | [0.67, 2.01] | 1.15 | [0.51, 1.86] | +0.18 [-0.31, +0.72] | 0.950 | 0.209 | 16.8% | 19.2% | -13.3% | -24.5% | 76.7% | 3 |
| regime_switch | 2020-04-08 | 1191 | 0.84 | [0.16, 1.69] | 1.15 | [0.51, 1.86] | -0.31 [-0.90, +0.33] | 0.732 | 0.031 | 9.5% | 19.2% | -19.1% | -24.5% | 28.7% | 26 |
| regime_parameters | 2020-04-08 | 1191 | 0.80 | [0.07, 1.50] | 1.15 | [0.51, 1.86] | -0.35 [-0.77, +0.05] | 0.697 | 0.027 | 10.2% | 19.2% | -24.7% | -24.5% | 78.3% | 21 |
| regime_sized | 2020-04-08 | 1191 | 1.26 | [0.61, 1.94] | 1.15 | [0.51, 1.86] | +0.11 [-0.42, +0.64] | 0.931 | 0.168 | 11.8% | 19.2% | -8.0% | -24.5% | 59.3% | 36 |
| adaptive_ensemble | 2020-04-08 | 1191 | 0.90 | [0.19, 1.71] | 1.15 | [0.51, 1.86] | -0.25 [-0.87, +0.38] | 0.776 | 0.042 | 7.2% | 19.2% | -13.6% | -24.5% | 21.7% | 278 |
| ml_regime_conditional | 2019-11-22 | 1284 | 0.74 | [0.11, 1.57] | 0.78 | [0.15, 1.60] | -0.04 [-0.14, +0.05] | 0.671 | 0.099 | 13.9% | 15.1% | -33.7% | -33.7% | 94.0% | 109 |

## Deflated Sharpe by trial count

n_trials in (1, 5, 11). 11 is every strategy in both tables above. Each strategy is scored on the window it appears in above.

| Strategy | Window | DSR vs 0 @1 | DSR vs 0 @5 | DSR vs 0 @11 | DSR vs bench @1 | DSR vs bench @5 | DSR vs bench @11 |
|---|---|---|---|---|---|---|---|
| sma_crossover | full period | 0.998 | 0.957 | 0.902 | 0.682 | 0.236 | 0.126 |
| adaptive_ensemble | 2020-04-08 to 2024-12-31 | 0.974 | 0.776 | 0.628 | 0.295 | 0.042 | 0.015 |

## Mean block length sensitivity

Net Sharpe 90% CI at n_boot=2000, for mean_block in (10, 20, 40). A CI that moves a lot with mean_block means the width itself isn't trustworthy at any single block length. Each strategy on the window it is scored on above.

| Strategy | mean_block=10 | mean_block=20 | mean_block=40 |
|---|---|---|---|
| sma_crossover | [0.33, 1.17] | [0.35, 1.17] | [0.34, 1.16] |
| adaptive_ensemble | [0.16, 1.71] | [0.19, 1.71] | [0.18, 1.75] |

## Passive-equivalent comparison: volatility_targeted

volatility_targeted's average exposure is 71.7%. Comparing it against a constant position held at that same exposure the whole period isolates the value of dynamic sizing from the trivial effect of just holding less.

| | Sharpe | Ann. Vol | Max DD |
|---|---|---|---|
| volatility_targeted (actual) | 0.832 | 11.0% | -18.7% |
| Passive @ 71.7% exposure | 0.607 | 14.3% | -39.7% |

## Rolling walk-forward: refit-per-fold vs fixed-model

train_days=756, test_days=126, cost_bps=5. Mean Sharpe across out-of-sample folds -- not the Sharpe in the tables above, and not affected by them either way. 'Fixed-model' reuses one fit across every fold (the old default); 'refit per fold' refits fresh on each fold's own training window, which is honest but n_folds times more expensive. Only folds where the strategy can hold a position on every test day are averaged; the fold columns count them. 'Change' is refit minus fixed-model over the same folds.

| Strategy | Fixed-model folds | Fixed-model mean fold Sharpe | Refit-per-fold, same folds | Change | Refit-per-fold folds | Refit-per-fold mean fold Sharpe |
|---|---|---|---|---|---|---|
| regime_switch | 8 of 27 | 1.29 | 0.89 | -0.40 | 27 of 27 | 0.86 |
| adaptive_ensemble | 8 of 27 | 1.21 | 0.84 | -0.36 | 27 of 27 | 0.79 |
| regime_filtered | 8 of 27 | 1.44 | 1.41 | -0.03 | 27 of 27 | 0.68 |
| sma_crossover | 27 of 27 | 1.08 | 1.08 | +0.00 | 27 of 27 | 1.08 |
| momentum | 27 of 27 | 0.85 | 0.85 | +0.00 | 27 of 27 | 0.85 |
| mean_reversion | 27 of 27 | 1.35 | 1.35 | +0.00 | 27 of 27 | 1.35 |
| volatility_targeted | 27 of 27 | 0.97 | 0.97 | +0.00 | 27 of 27 | 0.97 |
| ml_regime_conditional | 9 of 27 | 1.31 | 1.36 | +0.05 | 27 of 27 | 0.99 |
| regime_sized | 8 of 27 | 1.30 | 1.36 | +0.06 | 27 of 27 | 0.99 |
| ml_direction | 9 of 27 | 1.35 | 1.44 | +0.09 | 27 of 27 | 1.19 |
| regime_parameters | 8 of 27 | 0.73 | 1.29 | +0.56 | 27 of 27 | 0.96 |
