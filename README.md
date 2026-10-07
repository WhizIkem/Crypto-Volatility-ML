# Crypto Volatility Forecasting & Paper Trading System

An end-to-end quantitative finance and machine learning project for forecasting cryptocurrency volatility and using volatility forecast to drive dynamic portfolio allocation and paper trading.

## Project Goals

This project will:

- Collect and validate cryptocurrency market data.
- Perform exploratory data analysis.
- Engineer volatility and market features.
- Train and compare statistical and machine learning forecasting models.
- Evaluate models using time-series appropriate validation.
- Select the best-performing forecasting model.
- Construct a volatility-driven portfolio allocation strategy.
- Backtest the strategy with realistic trading costs.
- intergrate live cryptocurrency market data.
- Run a live paper-trading simulation.
- Implement automated testing, Docker, CI/CD, and monitoring.

## Assets

- Bitcoin (BTC)
- Ethereum (ETH)
- Solana (SOL)
- Binance Coin (BNB)

## Planned models
- Naive volatility baseline
- HAR-RV
- EGARCH
- Tree-based ML models
- LSTM
- Additional models where justified

## Project Status

Under active development.

### Current stage
**Phase 2 — Historical Data Pipeline**

Completed:

- ✅ Project repository and environment setup
- ✅ Installable Python package
- ✅ pytest and Ruff configuration
- ✅ Project configuration system
- ✅ Binance public API client
- ✅ Canonical candle normalisation
- ✅ Historical pagination
- ✅ Multi-asset data download for BTCUSDT, ETHUSDT, SOLUSDT, and BNBUSDT
- ✅ Closed-candle filtering

In progress:

- 🔄 Data validation
- 🔄 Missing timestamp / gap detection

Next:

- Reproducible dataset persistence
- Exploratory data analysis