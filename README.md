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

**Phase 5: Model-Ready Datasets**

### Completed

- ✅ Phase 1 — Project foundation and environment setup
- ✅ Phase 2 — Binance historical data pipeline
- ✅ Historical OHLCV download and pagination
- ✅ Data validation and missing-timestamp checks
- ✅ Raw and cleaned Parquet persistence
- ✅ Phase 3 — Exploratory data analysis
- ✅ Price, return, volatility, and distribution diagnostics
- ✅ Correlation, autocorrelation, ARCH, stationarity, and regime analysis
- ✅ Reusable EDA analysis modules
- ✅ Reproducible EDA tables and figures
- ✅ Phase 4 — Feature engineering
- ✅ Next-day log Garman–Klass volatility target
- ✅ Lagged return and volatility features
- ✅ HAR-style weekly and monthly volatility features
- ✅ Return-magnitude and downside-shock features
- ✅ Volume and trading-activity features
- ✅ Momentum and RSI features
- ✅ Cross-asset aggregate features
- ✅ Feature redundancy and missingness audit
- ✅ Reusable feature-engineering pipeline
- ✅ Model-ready dataset construction
- ✅ Unit tests for feature generation and dataset preparation
- ✅ Processed model datasets saved to Parquet

### In Progress

- 🔄 Phase 5 — Model-Ready Datasets

### Next

- Define chronological train / validation / test periods
- Prevent temporal leakage during splitting
- Decide whether scaling is needed per model family
- Fit preprocessing only on training data
- Save split metadata
- Verify feature and target alignment after splitting
- Prepare datasets for baseline and forecasting models