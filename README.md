# Crypto Volatility Forecasting & Paper Trading System

An end-to-end machine learning and quantitative finance project for forecasting cryptocurrency volatility, converting forecasts into risk-aware portfolio allocations, and running a production-style paper-trading system.

The project covers the full workflow from research to deployment-oriented infrastructure:

- historical Binance data ingestion;
- volatility feature engineering;
- econometric, ML, and deep-learning models;
- leakage-safe time-series validation;
- portfolio construction and backtesting;
- production model persistence and inference;
- live Binance data;
- persistent paper trading;
- runtime safety and reconciliation;
- Docker and CI/CD.

## Assets

- BTCUSDT
- ETHUSDT
- SOLUSDT
- BNBUSDT

Market data are sourced from the Binance public Spot API using closed 1-day candles.

## Forecasting Target

Features available at day `t` are used to predict:

**next-day log Garman-Klass volatility**

The research pipeline uses chronological splits, purged boundaries, training-only scaling, expanding-window validation, and a separate untouched final test period.

## Selected Models

Final model selection used four expanding validation windows.

| Asset | Model | Configuration | Robust RMSE |
| --- | --- | --- | ---: |
| BTC | VAR | lag = 7 | 0.4471 |
| ETH | LSTM | seq = 7, hidden = 32, lr = 0.0005 | 0.4231 |
| SOL | VAR | lag = 7 | 0.3889 |
| BNB | Ridge | alpha = 100 | 0.4482 |

Final test results:

| Asset | MAE | RMSE |
| --- | ---: | ---: |
| BTC | 0.3506 | 0.4483 |
| ETH | 0.3566 | 0.4597 |
| SOL | 0.3402 | 0.4413 |
| BNB | 0.3369 | 0.4250 |

Model choices and hyperparameters were frozen before the final test.

## Portfolio Strategy

Volatility forecasts are converted into long-only portfolio weights using:

```text
next-day volatility forecasts
        ↓
inverse-volatility allocation
        ↓
40% maximum asset weight
        ↓
30-day return correlation
        ↓
forecast covariance
        ↓
20% annual volatility target
        ↓
crypto exposure + cash
```

Execution rules:

- 20% annualized volatility target;
- 40% maximum weight per asset;
- 30-day correlation lookback;
- maximum risky exposure of 100%;
- no leverage;
- long-only;
- 10 bps transaction costs;
- rebalance only when one-way turnover is at least 5%.

The strategy is long-only because the models forecast risk, not expected return direction.

## Backtest Results

Final out-of-sample strategy results:

| Metric | Result |
| --- | ---: |
| Total return | +0.74% |
| Annualized volatility | 23.07% |
| Sharpe ratio | 0.152 |
| Maximum drawdown | -25.01% |
| Average risky exposure | 53.39% |
| Average cash weight | 46.61% |
| Rebalances | 206 |

Benchmark comparison:

| Strategy | Total Return | Ann. Volatility | Max Drawdown |
| --- | ---: | ---: | ---: |
| BTC Buy & Hold | -5.73% | 44.74% | -39.53% |
| Equal Weight Buy & Hold | -10.00% | 49.76% | -46.92% |
| Volatility Strategy | **+0.74%** | **23.07%** | **-25.01%** |

The main value of the strategy is risk reduction rather than directional return generation.

## Production Pipeline

Frozen model specifications are retrained on the full historical dataset and persisted for inference.

Production artifacts include:

- joint VAR model;
- VAR history state;
- ETH LSTM checkpoint;
- ETH feature and target scalers;
- BNB Ridge model;
- BNB feature scaler;
- production manifest.

The live pipeline uses the same feature-engineering logic as the research pipeline.

```text
Binance closed candles
        ↓
feature engineering
        ↓
production models
        ↓
volatility forecasts
        ↓
portfolio allocation
        ↓
paper-trading decision
```

## Paper Trading

The paper-trading system maintains a persistent `$1,000,000` simulated portfolio.

It supports:

- persistent cash and asset quantities;
- buy / sell / hold decisions;
- position sizing in asset units;
- transaction-cost deduction;
- 5% turnover threshold;
- duplicate-candle protection;
- portfolio and trade journaling;
- crash-recoverable persistence;
- startup integrity validation;
- single-process locking;
- accounting reconciliation.

Current runtime files:

```text
logs/paper_portfolio_state.json
logs/paper_portfolio_history.csv
logs/paper_trades.csv
logs/paper_trading.log
```

Runtime files are excluded from version control.

The current paper account has successfully processed multiple daily cycles, with state, equity, cash, and transaction costs reconciling across persisted state and trade journals.

## Runtime Safety

The production-style paper-trading path includes:

```text
process lock
    ↓
crash recovery
    ↓
startup validation
    ↓
duplicate-candle check
    ↓
forecast + portfolio decision
    ↓
atomic broker execution
    ↓
transactional persistence
    ↓
reconciliation
```

## Docker

The project runs inside a Linux Docker container.

The containerized environment has been validated with:

- Python 3.14;
- Ruff;
- 287 passing tests;
- production model loading;
- live Binance ingestion;
- paper trading against the existing account;
- paper-account reconciliation.

Common commands:

```bash
docker compose run --rm lint
docker compose run --rm test
docker compose run --rm paper-trader
docker compose run --rm reconcile
```

Production models are mounted read-only, while paper-trading state is persisted through the local `logs/` directory.

## CI/CD

GitHub Actions builds the Docker image and automatically runs:

```text
Ruff
pytest
```

on pushes and pull requests.

The CI workflow does not run the live paper trader or access runtime account state.

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
notebooks/
reports/
docker/
```

Reusable logic lives in `src/`, while notebooks are retained for research and analysis.

## Current Status

Completed:

- ✅ historical data pipeline
- ✅ EDA and feature engineering
- ✅ econometric / ML / DL model research
- ✅ robust time-series validation
- ✅ final model selection
- ✅ portfolio construction
- ✅ realistic backtesting
- ✅ production model training and inference
- ✅ live Binance pipeline
- ✅ persistent paper trading
- ✅ execution safety and reconciliation
- ✅ Docker
- ✅ Docker Compose
- ✅ automated testing
- ✅ GitHub Actions CI
- ✅ Oracle Cloud deployment
- ✅ ARM64 production Docker runtime
- ✅ automated daily paper-trading execution with systemd
- ✅ cloud-hosted reconciliation and runtime validation

Next:

- 🔄 monitoring and observability
- ⬜ P&L and performance monitoring
- ⬜ Streamlit dashboard
- ⬜ final project documentation and portfolio presentation

## Key Finding

The project shows that volatility forecasting can be used effectively as a **risk-management layer**.

Different assets benefit from different forecasting models, and combining those forecasts with volatility targeting, cash allocation, concentration limits, and turnover controls materially reduces portfolio risk relative to fully invested crypto benchmarks.