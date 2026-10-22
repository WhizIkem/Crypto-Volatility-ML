# Crypto Volatility Forecasting & Paper Trading System

An end-to-end quantitative finance and machine learning project for forecasting cryptocurrency volatility and using those forecasts for portfolio allocation, backtesting, and live paper trading.

## Project Goals

This project aims to:

- collect and validate cryptocurrency market data;
- explore price, return, and volatility behaviour;
- engineer useful volatility and market features;
- compare statistical, econometric, machine-learning, and deep-learning models;
- use leakage-safe time-series validation;
- select the strongest forecasting models;
- build a volatility-driven portfolio strategy;
- backtest with realistic trading costs;
- connect to live Binance market data;
- run paper-trading simulations;
- add testing, Docker, CI/CD, logging, and monitoring.

## Assets

- Bitcoin — BTCUSDT
- Ethereum — ETHUSDT
- Solana — SOLUSDT
- Binance Coin — BNBUSDT

## Data

Historical market data are collected from the Binance public Spot API.

Current setup:

- Interval: 1 day
- Quote asset: USDT
- Start date: 2021-01-01
- Closed candles only
- Storage format: Parquet

## Forecast Target

The main forecasting target is:

**next-day log Garman-Klass volatility**

Features available at day `t` are used to predict volatility at day `t + 1`.

## Feature Set

A broad research pool of 28 candidate features was created.

After a training-only feature audit, the main ML feature set was reduced to seven interpretable predictors:

- current log Garman-Klass volatility;
- 7-day average volatility;
- 30-day average volatility;
- absolute return;
- log volume;
- cross-asset mean log volatility;
- downside return.

The full 28-feature set is still kept for research comparisons.

## Leakage Prevention

Time-series leakage is treated as a hard constraint.

The modelling pipeline uses:

- chronological train / validation / test splits;
- no random shuffling of time-series partitions;
- a one-row purge at split boundaries for the one-day-ahead target;
- training-only feature scaling;
- training-only target scaling for LSTM models;
- expanding-window time-series cross-validation;
- one-row gaps inside time-series CV folds;
- features built only from information available at forecast time;
- an untouched final test set.

Purged rows are excluded as supervised targets when their `t + 1` label would cross a split boundary.

Their observed market features may still be used as legitimate historical context for later sequence-model predictions.

The final test set will only be used after model selection and tuning are complete.

## Model Results

### Naive Baseline

The 7-day rolling mean was the strongest naive benchmark.

| Asset | Validation RMSE |
| --- | ---: |
| BTC | 0.5086 |
| ETH | 0.4865 |
| SOL | 0.4204 |
| BNB | 0.4829 |

### HAR-GK

HAR-GK uses current, 7-day, and 30-day log Garman-Klass volatility.

It beat the naive baseline for all four assets.

| Asset | HAR-GK RMSE |
| --- | ---: |
| BTC | 0.4904 |
| ETH | 0.4553 |
| SOL | 0.3968 |
| BNB | 0.4538 |

The 7-day volatility component received the largest HAR coefficient across all assets.

### EGARCH

An EGARCH(1,1,1) model with Student-t innovations was evaluated using expanding-window forecasts.

| Asset | EGARCH RMSE |
| --- | ---: |
| BTC | 0.6354 |
| ETH | 0.5283 |
| SOL | 0.4859 |
| BNB | 0.5656 |

EGARCH underperformed both HAR-GK and the 7-day rolling benchmark.

For this target, range-based volatility persistence appears more useful than return-only conditional variance dynamics.

### Ridge Regression

Ridge regression used the compact seven-feature set.

The regularization parameter was selected with training-only time-series cross-validation.

| Asset | Best Alpha | Validation RMSE |
| --- | ---: | ---: |
| BTC | 100 | 0.4886 |
| ETH | 10 | 0.4470 |
| SOL | 0.001 | 0.3941 |
| BNB | 100 | 0.4512 |

Ridge produced modest improvements over HAR-GK while remaining interpretable.

### Lasso Regression

Lasso used the same seven-feature set and validation protocol.

| Asset | Validation RMSE |
| --- | ---: |
| BTC | 0.4899 |
| ETH | 0.4468 |
| SOL | 0.3943 |
| BNB | 0.4497 |

Ridge and Lasso performed similarly.

Lasso removed only one feature completely: BTC log volume.

The 7-day volatility feature remained the strongest standardized coefficient across all assets.

### Nonlinear ML

Random Forest and Gradient Boosting were evaluated using the compact seven-feature set and training-only time-series hyperparameter tuning.

| Asset | Ridge | Lasso | Random Forest | Gradient Boosting |
| --- | ---: | ---: | ---: | ---: |
| BTC | 0.4886 | 0.4899 | **0.4865** | 0.4974 |
| ETH | 0.4470 | **0.4468** | 0.4656 | 0.4499 |
| SOL | **0.3941** | 0.3943 | 0.3976 | 0.3942 |
| BNB | 0.4512 | **0.4497** | 0.4630 | 0.4646 |

Nonlinear tree models did not consistently outperform the simpler regularized linear models.

Random Forest performed best for BTC, while Ridge and Lasso remained stronger for the other assets.

### LSTM

The first LSTM experiment was followed by a training audit to make the neural-network comparison more robust.

The revised LSTM pipeline added:

- training-only target standardization;
- shuffled independent training sequences;
- sequence-length tuning;
- hidden-size tuning;
- learning-rate tuning;
- early stopping;
- evaluation across three fixed random seeds.

The final test set remained untouched.

| Asset | Sequence Length | Hidden Size | Mean RMSE | RMSE Std |
| --- | ---: | ---: | ---: | ---: |
| BTC | 30 | 16 | 0.4874 | 0.0037 |
| ETH | 7 | 32 | **0.4335** | 0.0058 |
| SOL | 7 | 32 | 0.3947 | 0.0013 |
| BNB | 7 | 32 | 0.4685 | 0.0070 |

The tuned LSTM materially improved ETH forecasting and became competitive with Random Forest for BTC.

For SOL and BNB, simpler regularized linear models remained stronger.

## Current Model Leaders

| Asset | Best Validation Model | RMSE |
| --- | --- | ---: |
| BTC | Random Forest | 0.4865 |
| ETH | LSTM | 0.4335 mean |
| SOL | Ridge | 0.3941 |
| BNB | Lasso | 0.4497 |

These are validation results only.

The final test set remains untouched.

## Current Findings

So far:

1. Cryptocurrency volatility is strongly persistent.
2. The 7-day volatility measure is the most consistent predictor.
3. HAR-GK clearly improves on naive volatility baselines.
4. EGARCH is weaker for the selected range-based target.
5. Ridge and Lasso provide small but consistent improvements over HAR-GK.
6. More features do not automatically improve forecasting performance.
7. Nonlinear tree models help selectively rather than universally.
8. LSTM performance is highly dependent on training design and optimization.
9. The tuned LSTM is strongest for ETH and competitive for BTC.
10. Simpler models remain stronger for SOL and BNB.
11. The compact seven-feature set captures most of the useful signal while remaining interpretable.

## Project Status

**Current phase: Phase 6 — Model Research**

Completed:

- ✅ Project setup
- ✅ Binance historical data pipeline
- ✅ Exploratory data analysis
- ✅ Feature engineering
- ✅ Model-ready datasets
- ✅ Leakage-safe time-series splitting
- ✅ Naive volatility baselines
- ✅ HAR-GK
- ✅ EGARCH
- ✅ Training-only feature audit
- ✅ Compact feature selection
- ✅ Ridge regression
- ✅ Lasso regression
- ✅ Random Forest
- ✅ Gradient Boosting
- ✅ LSTM sequence modelling
- ✅ LSTM training audit
- ✅ Training-only hyperparameter tuning

Next:

- 🔄 VAR / cross-asset volatility analysis
- ⬜ Robust model comparison
- ⬜ Final untouched test evaluation
- ⬜ Portfolio allocation
- ⬜ Backtesting
- ⬜ Live Binance inference
- ⬜ Paper trading
- ⬜ Deployment and monitoring

## Model Families

Current and planned models include:

- naive volatility baselines;
- HAR-GK;
- EGARCH;
- Ridge;
- Lasso;
- Random Forest;
- Gradient Boosting;
- LSTM;
- VAR / cross-asset volatility analysis.

## Repository Structure

```text
src/crypto_volatility/
├── data/
├── analysis/
├── features/
├── models/
├── evaluation/
├── portfolio/
├── trading/
└── utils/

scripts/
tests/
reports/
notebooks/
```

Research notebooks are kept as experiment records, while reusable and tested logic is moved into `src/`.