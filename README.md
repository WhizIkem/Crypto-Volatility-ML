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

**Phase 4: Feature Engineering**

### Completed

- ✅ Phase 1 — Project foundation and environment setup
- ✅ Phase 2 — Binance historical data pipeline
- ✅ Historical OHLCV download and pagination
- ✅ Data validation and missing-timestamp checks
- ✅ Raw and cleaned Parquet persistence
- ✅ Phase 3 — Exploratory data analysis
- ✅ Price and normalized-price analysis
- ✅ Log-return analysis
- ✅ Rolling and Garman–Klass volatility analysis
- ✅ Return distribution and normality diagnostics
- ✅ Cross-asset return and volatility correlations
- ✅ Rolling correlation analysis
- ✅ ACF and PACF diagnostics
- ✅ ARCH-LM testing
- ✅ ADF and KPSS stationarity diagnostics
- ✅ Log-volatility analysis
- ✅ Volatility regime analysis
- ✅ Reusable EDA analysis modules
- ✅ Unit tests for core EDA calculations and diagnostics
- ✅ Reproducible EDA pipeline
- ✅ Reproducible EDA tables and selected figures

### In Progress

- 🔄 Phase 4 — Feature Engineering

### Next

- Define the forecasting target
- Build lagged return features
- Build lagged volatility features
- Build HAR-style daily, weekly, and monthly volatility features
- Build squared-return and downside-return features
- Build volume and trading-activity features
- Investigate RSI and other candidate market features
- Prevent temporal leakage
- Validate feature availability at prediction time
- Produce the model-ready dataset