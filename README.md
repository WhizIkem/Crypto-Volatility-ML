# Crypto Volatility Forecasting & Paper Trading System

An end-to-end quantitative finance and machine learning project for forecasting cryptocurrency volatility and using those forecasts for portfolio allocation, backtesting, and live paper trading.

## Project Goals

This project aims to:

- collect and validate cryptocurrency market data;
- explore price, return, and volatility behaviour;
- engineer useful volatility and market features;
- compare statistical, econometric, ML, and deep-learning models;
- use leakage-safe time-series validation;
- select the best forecasting model;
- build a volatility-based portfolio strategy;
- backtest with realistic trading costs;
- connect to live Binance data;
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

## Current Feature Set

A broad set of 28 candidate features was created during research.

After a training-only feature audit, the main ML feature set was reduced to seven interpretable features:

- current log Garman-Klass volatility;
- 7-day average volatility;
- 30-day average volatility;
- absolute return;
- log volume;
- cross-asset mean log volatility;
- downside return.

The 28-feature set is still kept for research comparisons.

## Leakage Prevention

Time-series leakage is treated as a hard constraint.

The pipeline uses:

- chronological train / validation / test splits;
- no random shuffling;
- a one-row purge at split boundaries for the one-day-ahead target;
- training-only feature scaling;
- expanding-window time-series cross-validation;
- a one-row gap inside cross-validation folds;
- an untouched final test set.

The final test set will only be used after feature selection, model selection, and hyperparameter tuning are complete.

## Model Results So Far

### Naive Baselines

The 7-day rolling mean was the strongest naive volatility benchmark for all four assets.

| Asset | Validation RMSE |
| --- | ---: |
| BTC | 0.5086 |
| ETH | 0.4865 |
| SOL | 0.4204 |
| BNB | 0.4829 |

### HAR-GK

HAR-GK uses current, 7-day, and 30-day log Garman-Klass volatility.

It beat the 7-day rolling baseline for all four assets.

| Asset | HAR-GK RMSE |
| --- | ---: |
| BTC | 0.4904 |
| ETH | 0.4553 |
| SOL | 0.3968 |
| BNB | 0.4538 |

The 7-day volatility component had the largest HAR coefficient for every asset.

### EGARCH

An EGARCH(1,1,1) model with Student-t errors was evaluated using expanding-window forecasts.

| Asset | EGARCH RMSE |
| --- | ---: |
| BTC | 0.6354 |
| ETH | 0.5283 |
| SOL | 0.4859 |
| BNB | 0.5656 |

EGARCH underperformed both HAR-GK and the 7-day baseline for this forecasting target.

This suggests that range-based volatility persistence is more useful here than return-only conditional variance modelling.

### Ridge Regression

Ridge regression was trained using the compact seven-feature set.

The regularization parameter was selected using training-only expanding-window cross-validation.

| Asset | Best Alpha | Validation RMSE |
| --- | ---: | ---: |
| BTC | 100 | 0.4886 |
| ETH | 10 | 0.4470 |
| SOL | 0.001 | 0.3941 |
| BNB | 100 | 0.4512 |

Ridge slightly improved on HAR-GK while keeping the model interpretable.

### Lasso Regression

Lasso was tested using the same seven-feature set and the same leakage-safe validation process.

| Asset | Validation RMSE |
| --- | ---: |
| BTC | 0.4899 |
| ETH | 0.4468 |
| SOL | 0.3943 |
| BNB | 0.4497 |

Ridge and Lasso performed very similarly.

Lasso removed only one feature completely: BTC log volume.

The 7-day volatility feature remained the strongest standardized coefficient across all assets.

## Current Findings

So far:

1. Volatility is strongly persistent.
2. The 7-day volatility measure is the most consistent predictor.
3. HAR-GK beats simple volatility baselines.
4. EGARCH is weaker for the selected Garman-Klass target.
5. Ridge and Lasso provide small improvements over HAR-GK.
6. A compact seven-feature set performs well without using all 28 candidate features.

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
- ✅ Training-only time-series hyperparameter tuning

Next:

- 🔄 Nonlinear ML models
- ⬜ Deep-learning models
- ⬜ Final validation comparison
- ⬜ Final untouched test evaluation
- ⬜ Portfolio allocation
- ⬜ Backtesting
- ⬜ Live Binance inference
- ⬜ Paper trading
- ⬜ Deployment and monitoring

## Planned Model Families

The project compares models only where they are useful and justified.

Current and planned models include:

- naive volatility baselines;
- HAR-GK;
- EGARCH;
- Ridge;
- Lasso;
- tree-based ML models;
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