# Crypto Volatility Forecasting & Paper Trading System

An end-to-end machine learning and quantitative finance system for forecasting cryptocurrency volatility, translating forecasts into risk-aware portfolio allocations, and running an automated cloud-hosted paper-trading workflow.

The project covers the complete path from research to production-style deployment:

- historical Binance data ingestion;
- volatility feature engineering;
- econometric, machine-learning, and deep-learning models;
- leakage-safe time-series validation;
- portfolio construction and realistic backtesting;
- production model persistence and inference;
- live Binance market data;
- persistent paper trading;
- execution safety and reconciliation;
- Dockerized production runtimes;
- automated testing and GitHub Actions CI;
- Oracle Cloud deployment;
- automated daily execution;
- runtime monitoring and observability;
- Streamlit production dashboard.

## Assets

- BTCUSDT
- ETHUSDT
- SOLUSDT
- BNBUSDT

Market data are sourced from the Binance public Spot API using closed 1-day candles.

## Forecasting Target

Features available at day `t` are used to predict:

**next-day log Garman-Klass volatility**

The research pipeline uses:

- chronological train / validation / test splits;
- purged split boundaries;
- training-only feature scaling;
- expanding-window validation;
- a separate untouched final test period.

## Selected Models

Final model selection used four expanding validation windows.

| Asset | Model | Configuration | Robust RMSE |
| --- | --- | --- | ---: |
| BTC | VAR | lag = 7 | 0.4471 |
| ETH | LSTM | seq = 7, hidden = 32, lr = 0.0005 | 0.4231 |
| SOL | VAR | lag = 7 | 0.3889 |
| BNB | Ridge | alpha = 100 | 0.4482 |

Final untouched test results:

| Asset | MAE | RMSE |
| --- | ---: | ---: |
| BTC | 0.3506 | 0.4483 |
| ETH | 0.3566 | 0.4597 |
| SOL | 0.3402 | 0.4413 |
| BNB | 0.3369 | 0.4250 |

Model choices and hyperparameters were frozen before evaluation on the final test period.

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
- long-only allocation;
- 10 bps transaction costs;
- rebalance only when one-way turnover is at least 5%.

The strategy is long-only because the forecasting models estimate risk rather than expected return direction.

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
| **Volatility Strategy** | **+0.74%** | **23.07%** | **-25.01%** |

The primary value of the strategy is **risk reduction and capital preservation**, rather than aggressive directional return generation.

## Production Pipeline

Frozen model specifications are retrained on the full historical dataset and persisted for production inference.

Production artifacts include:

- joint VAR model;
- VAR history state;
- ETH LSTM checkpoint;
- ETH feature scaler;
- ETH target scaler;
- BNB Ridge model;
- BNB feature scaler;
- production manifest.

The live pipeline reuses the same feature-engineering logic as the research pipeline.

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

The paper-trading system maintains a persistent simulated portfolio initialized with:

```text
$1,000,000
```

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

Runtime files include:

```text
logs/paper_portfolio_state.json
logs/paper_portfolio_history.csv
logs/paper_trades.csv
logs/paper_trading.log
logs/paper_scheduler.log
```

Runtime files are excluded from version control.

The paper account persists across daily runs and validates state, equity, cash, trade history, and transaction costs after execution.

## Runtime Safety

The production paper-trading path includes:

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

Additional safeguards include:

- prevention of concurrent paper-trading processes;
- duplicate forecast-origin protection;
- persisted-state validation before execution;
- atomic state and journal persistence;
- post-cycle accounting reconciliation;
- scheduler-health monitoring;
- stale-runtime detection.

## Monitoring & Observability

A dedicated read-only monitoring layer tracks the running paper account.

Current monitoring includes:

- current equity;
- cumulative P&L;
- cumulative return;
- latest-period P&L and return;
- equity curve;
- current and maximum drawdown;
- risky exposure and cash allocation;
- current asset weights;
- transaction costs;
- trade and rebalance counts;
- latest forecast origin;
- latest trading run;
- scheduler freshness;
- reconciliation health;
- runtime integrity checks.

Annualized volatility and Sharpe ratio are displayed only after a minimum of 30 daily portfolio observations to avoid presenting highly unstable statistics from an extremely small sample.

## Streamlit Dashboard

The project includes a read-only Streamlit dashboard for production monitoring.

The dashboard provides four main views:

### Overview

- current equity;
- cumulative P&L and return;
- risky exposure and cash reserve;
- latest-period performance;
- drawdown;
- transaction costs;
- equity curve;
- current allocation;
- risk-history readiness.

### Portfolio & Forecasts

- latest volatility forecasts;
- current vs target portfolio allocation;
- weight changes;
- rebalance signal;
- risky and cash exposure;
- portfolio decision summary.

### Execution

- trading cycles;
- execution rate;
- portfolio turnover;
- transaction-cost history;
- turnover threshold;
- recent simulated trades.

### System Health

- scheduler health;
- reconciliation status;
- state/history consistency;
- transaction-cost consistency;
- duplicate-origin checks;
- runtime timeline;
- operational safeguards.

The dashboard does not execute trades or modify portfolio state.

## Oracle Cloud Deployment

The production runtime is deployed on an Oracle Cloud ARM64 virtual machine.

Environment:

```text
Architecture: aarch64
Memory: ~12 GB
Storage: ~45 GB
Operating system: Ubuntu
```

The deployment uses a lean ARM64-compatible production Docker image with CPU-only PyTorch.

The Oracle VM hosts:

```text
production model runtime
        +
live Binance pipeline
        +
paper trader
        +
reconciliation
        +
monitoring
        +
Streamlit dashboard
```

Production models are mounted read-only while runtime paper-trading state is persisted separately.

## Automated Daily Execution

Paper trading is automatically triggered using a systemd timer on the Oracle VM.

```text
systemd timer
      ↓
scheduled wrapper
      ↓
Docker paper trader
      ↓
reconciliation
      ↓
scheduler health log
```

The scheduler:

- runs daily;
- persists across SSH logout;
- records successful and failed executions;
- runs reconciliation after every trading cycle.

The local development-machine scheduler is disabled so Oracle remains the single active production paper-trading host.

## Docker

The project uses separate Docker environments for development, production trading, and dashboard deployment.

### Development

Used for:

- Ruff;
- pytest;
- research/runtime validation.

### Production Trader

Uses a lean ARM64-compatible production image with CPU-only PyTorch.

### Dashboard

Uses a separate lightweight Streamlit image and mounts runtime logs read-only.

Example commands:

```bash
docker compose run --rm lint
docker compose run --rm test
docker compose run --rm paper-trader
docker compose run --rm reconcile
```

The production environment has been validated with:

- Python 3.14;
- Ruff;
- full automated test suite;
- production model loading;
- live Binance ingestion;
- paper trading against the persisted account;
- reconciliation;
- ARM64 Docker execution;
- Oracle Cloud deployment.

## Continuous Integration

GitHub Actions builds and validates the project on pushes and pull requests.

The CI workflow runs:

```text
Docker build
    ↓
Ruff
    ↓
pytest
```

The CI environment does **not** run the live paper trader, access production paper-account state, or execute scheduled cloud workloads.

## Repository Structure

```text
crypto-volatility-ml/
├── configs/
├── dashboard/
├── data/
├── docker/
├── logs/
├── models/
├── notebooks/
├── reports/
├── scripts/
├── src/
│   └── crypto_volatility/
│       ├── analysis/
│       ├── data/
│       ├── evaluation/
│       ├── features/
│       ├── models/
│       ├── monitoring/
│       ├── portfolio/
│       ├── trading/
│       └── utils/
├── tests/
├── requirements.txt
├── requirements-prod.txt
├── requirements-dashboard.txt
├── compose.yaml
└── README.md
```

Reusable application logic lives in `src/`, notebooks are retained for research and analysis, and production/runtime concerns are separated from model experimentation.

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
- ✅ monitoring and observability
- ✅ runtime health and scheduler monitoring
- ✅ paper-trading performance and P&L monitoring
- ✅ Streamlit production dashboard

Next:

- 🔄 final dashboard validation on Oracle
- ⬜ final project documentation and cleanup

## Key Finding

This project demonstrates that volatility forecasting can be used effectively as a **risk-management layer** rather than simply as a prediction exercise.

Different assets benefit from different forecasting models, and combining those forecasts with volatility targeting, correlation-aware allocation, cash reserves, concentration limits, transaction costs, and turnover controls materially reduces portfolio risk relative to fully invested crypto benchmarks.

The project also demonstrates the complete engineering path from:

```text
research
    ↓
model validation
    ↓
portfolio construction
    ↓
backtesting
    ↓
production inference
    ↓
live data
    ↓
automated paper trading
    ↓
cloud deployment
    ↓
monitoring and observability
```