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

Under active development.

### Current Stage

**Phase 3: Exploratory Data Analysis**

### Completed

- ✅ Project repository and environment setup
- ✅ Installable Python package
- ✅ pytest and Ruff configuration
- ✅ Project configuration system
- ✅ Binance public API client
- ✅ Canonical candle normalisation
- ✅ Historical pagination
- ✅ Multi-asset data download
- ✅ Closed-candle filtering
- ✅ Data validation
- ✅ Missing timestamp / gap detection
- ✅ Raw dataset persistence
- ✅ Clean/interim dataset persistence
- ✅ Reproducible Binance historical data pipeline

### In Progress

- 🔄 Exploratory data analysis

### Next

- Price behaviour
- Return analysis
- Volatility analysis
- Distribution diagnostics
- Correlation analysis
- Volatility clustering
- Regime behaviour
- Statistical diagnostics
- Feature engineering