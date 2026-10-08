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

**Phase 4: Feature Engineering**

### Completed

- ✅ Project foundation
- ✅ Binance historical data pipeline
- ✅ Data validation and persistence
- ✅ Exploratory data analysis
- ✅ Price and return analysis
- ✅ Volatility analysis
- ✅ Distribution diagnostics
- ✅ Correlation and rolling-correlation analysis
- ✅ Volatility clustering and autocorrelation analysis
- ✅ ARCH-LM diagnostics
- ✅ Stationarity diagnostics
- ✅ Volatility regime analysis

### In Progress

- 🔄 Feature engineering

### Next

- Log-return features
- Garman–Klass volatility features
- Lagged volatility
- Weekly and monthly volatility aggregates
- Negative-return and squared-return features
- Volume features
- RSI
- Binance-specific activity features
- Target construction
- Leakage checks