# Crypto Volatility Forecasting & Paper Trading System

An end-to-end quantitative finance and machine learning project for forecasting cryptocurrency volatility and using those forecasts to drive dynamic portfolio allocation, backtesting, and live paper trading.

## Project Goals

This project will:

- Collect and validate cryptocurrency market data.
- Perform exploratory data analysis.
- Engineer volatility and market features.
- Train and compare statistical, econometric, machine-learning, and deep-learning forecasting models.
- Evaluate models using time-series appropriate validation.
- Select the best-performing forecasting model.
- Construct a volatility-driven portfolio allocation strategy.
- Backtest the strategy with realistic trading costs.
- Integrate live cryptocurrency market data.
- Run a live paper-trading simulation.
- Implement automated testing, Docker, CI/CD, logging, and monitoring.

## Assets

- Bitcoin (BTC)
- Ethereum (ETH)
- Solana (SOL)
- Binance Coin (BNB)

Trading pairs:

- BTCUSDT
- ETHUSDT
- SOLUSDT
- BNBUSDT

## Planned Models

- Naive volatility baseline
- HAR-RV
- EGARCH
- Tree-based ML models
- LSTM
- VAR / volatility spillover analysis
- Additional models where justified

## Data Source

Historical market data are collected from the Binance public Spot API.

Current research configuration:

- Interval: 1 day
- Quote asset: USDT
- Start date: 2021-01-01
- Closed candles only
- Storage format: Parquet

## Project Status

### Current Stage

**Phase 6: Model Research**

### Completed

- ✅ Phase 1 — Project foundation and environment setup
- ✅ Phase 2 — Binance historical data pipeline
- ✅ Phase 3 — Exploratory data analysis
- ✅ Phase 4 — Feature engineering
- ✅ Phase 5 — Model-ready datasets
- ✅ Chronological train / validation / test splitting
- ✅ Leakage-safe time-series partitioning
- ✅ Train-only feature scaling
- ✅ Model input preparation
- ✅ Unit tests for preprocessing and model inputs

### In Progress

- 🔄 Phase 6 — Model Research

### Next

- Naive volatility baselines
- Historical-average baseline
- HAR-style regression
- EGARCH
- Linear ML baselines
- Tree-based ML models
- LSTM
- VAR / cross-asset volatility modelling
- Walk-forward validation
- Out-of-sample model comparison