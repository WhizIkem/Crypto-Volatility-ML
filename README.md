# Crypto Volatility Forecasting & Paper Trading System

An end-to-end machine learning and quantitative finance system for forecasting cryptocurrency volatility, converting forecasts into risk-aware portfolio allocations, and running an automated cloud-hosted paper-trading workflow.

## Live Dashboard

**Public dashboard:**  
https://cryptovolatility.duckdns.org

The dashboard provides live visibility into portfolio state, volatility forecasts, allocation decisions, execution activity, transaction costs, drawdown, scheduler health, reconciliation, and runtime integrity.

---

## Project Overview

The project covers the full workflow from research to production-style deployment:

- historical Binance data ingestion;
- exploratory analysis and volatility feature engineering;
- econometric, machine-learning, and deep-learning models;
- leakage-safe time-series validation;
- portfolio construction and realistic backtesting;
- production model persistence and inference;
- live Binance market data;
- automated persistent paper trading;
- runtime safety and accounting reconciliation;
- Dockerized execution;
- automated testing and GitHub Actions CI;
- Oracle Cloud deployment;
- automated daily execution with systemd;
- monitoring and observability;
- public Streamlit dashboard over HTTPS.

## Assets

The system models:

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
- training-only scaling;
- expanding-window validation;
- a separate untouched final test period.

Random train/test splitting is not used.

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

Model choices and hyperparameters were frozen before final test evaluation.

## Portfolio Strategy

Volatility forecasts are converted into long-only portfolio weights through:

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

Strategy rules:

- 20% annualized volatility target;
- 40% maximum weight per asset;
- 30-day correlation lookback;
- maximum risky exposure of 100%;
- no leverage;
- long-only allocation;
- 10 bps transaction costs;
- rebalance only when one-way turnover is at least 5%.

The strategy is long-only because the models forecast risk rather than expected return direction.

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

The strategy is designed primarily as a **risk-management layer**, not as a directional return-maximization strategy.

## Production Pipeline

Frozen model specifications are retrained on the full historical dataset and persisted for live inference.

Production artifacts include:

- joint VAR model;
- VAR history state;
- ETH LSTM checkpoint;
- ETH feature and target scalers;
- BNB Ridge model;
- BNB feature scaler;
- production manifest.

The live path reuses the same feature-engineering logic as the research pipeline.

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

The system maintains a persistent simulated portfolio initialized with:

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
- trade and portfolio journals;
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

Runtime state is excluded from version control.

## Runtime Safety

The paper-trading execution path includes:

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

Safeguards include:

- duplicate forecast-origin protection;
- prevention of concurrent execution;
- persisted-state validation;
- atomic state updates;
- crash recovery;
- post-cycle reconciliation;
- scheduler freshness checks;
- stale-runtime detection.

## Monitoring & Observability

A read-only monitoring layer tracks the running paper account.

It includes:

- current equity;
- cumulative P&L and return;
- latest-period P&L and return;
- equity curve;
- current and maximum drawdown;
- risky exposure and cash allocation;
- asset weights;
- transaction costs;
- trade and rebalance counts;
- latest forecast origin;
- latest trading run;
- scheduler freshness;
- reconciliation health;
- state/history consistency;
- duplicate-origin checks.

Annualized volatility and Sharpe ratio are displayed only after at least 30 daily portfolio observations are available.

## Streamlit Dashboard

The public monitoring interface is built with Streamlit.

**Live URL:**  
https://cryptovolatility.duckdns.org

The dashboard has four views:

### Overview

- portfolio equity;
- cumulative return;
- risky exposure;
- cash reserve;
- latest-period P&L;
- drawdown;
- transaction costs;
- equity curve;
- current allocation;
- risk-history readiness.

### Portfolio & Forecasts

- latest volatility forecasts;
- current vs target allocation;
- weight changes;
- rebalance signal;
- cash and risky exposure;
- asset-level allocation decisions.

### Execution

- trading cycles;
- execution rate;
- portfolio turnover;
- transaction costs;
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

The dashboard is read-only and cannot execute trades or modify portfolio state.

## Cloud Deployment

The production system runs on an Oracle Cloud ARM64 virtual machine.

```text
Architecture: aarch64
Operating system: Ubuntu
Memory: ~12 GB
Storage: ~45 GB
```

The Oracle VM hosts:

```text
production models
        +
live Binance pipeline
        +
paper trader
        +
persistent runtime state
        +
reconciliation
        +
monitoring
        +
Streamlit dashboard
        +
Caddy reverse proxy
```

The production runtime uses a lean ARM64-compatible Docker image with CPU-only PyTorch.

## Automated Daily Execution

Paper trading is triggered automatically using a systemd user timer on Oracle.

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

The production schedule runs daily at:

```text
00:15 UTC
```

Oracle Cloud is the single active scheduled paper-trading host.

## Public HTTPS

The Streamlit application runs internally on port `8501`.

Public traffic follows:

```text
Internet
   ↓
HTTPS
   ↓
cryptovolatility.duckdns.org
   ↓
Caddy
   ↓
Streamlit
   ↓
live Oracle runtime data
```

Caddy handles the reverse proxy and HTTPS, while DuckDNS provides the public hostname.

## Docker

The project uses separate Docker environments for development, production trading, and dashboard serving.

Common commands:

```bash
docker compose run --rm lint
docker compose run --rm test
docker compose run --rm paper-trader
docker compose run --rm reconcile
```

The production environment has been validated with:

- Python 3.14;
- Ruff;
- the full automated test suite;
- production model loading;
- live Binance ingestion;
- persistent paper trading;
- reconciliation;
- ARM64 Docker execution;
- Oracle Cloud execution;
- automated scheduling;
- public dashboard serving.

## Continuous Integration

GitHub Actions validates the project on pushes and pull requests.

The CI workflow runs:

```text
Docker build
    ↓
Ruff
    ↓
pytest
```

The CI environment does not:

- execute the live paper trader;
- access production account state;
- run the Oracle scheduler;
- deploy automatically to Oracle.

Production deployment is currently manual after validated changes are pushed to GitHub.

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

Reusable application logic lives in `src/`, while notebooks are retained for research and analysis.

## Running the Project

### Development checks

```bash
docker compose run --rm lint
docker compose run --rm test
```

### Run a paper-trading cycle manually

```bash
docker compose run --rm paper-trader
```

### Run reconciliation

```bash
docker compose run --rm reconcile
```

### Run the dashboard locally

Build:

```bash
docker build \
  -f docker/Dockerfile.dashboard \
  -t crypto-volatility:dashboard-dev \
  .
```

Run:

```bash
docker run --rm \
  -p 8502:8501 \
  -v "$PWD/logs:/app/logs:ro" \
  crypto-volatility:dashboard-dev
```

Then open:

```text
http://localhost:8502
```

## Current Status

Completed:

- ✅ historical Binance data pipeline
- ✅ EDA and feature engineering
- ✅ econometric / ML / DL model research
- ✅ leakage-safe time-series validation
- ✅ final model selection
- ✅ portfolio construction
- ✅ realistic backtesting
- ✅ production training and inference
- ✅ live Binance pipeline
- ✅ persistent paper trading
- ✅ transaction-cost modelling
- ✅ execution safety and crash recovery
- ✅ reconciliation
- ✅ Docker and Docker Compose
- ✅ automated testing
- ✅ GitHub Actions CI
- ✅ Oracle Cloud deployment
- ✅ ARM64 production runtime
- ✅ automated daily execution with systemd
- ✅ monitoring and observability
- ✅ Streamlit production dashboard
- ✅ public HTTPS deployment

### Live System

**Dashboard:**  
https://cryptovolatility.duckdns.org

**Trading schedule:**  
Daily at 00:15 UTC

**Runtime:**  
Oracle Cloud ARM64

**Mode:**  
Simulated paper trading

## Key Finding

The project demonstrates that volatility forecasting can be used effectively as a **risk-management layer**.

Different assets benefit from different forecasting approaches, and combining those forecasts with volatility targeting, correlation-aware allocation, cash reserves, concentration limits, transaction costs, and turnover controls materially reduces portfolio risk relative to fully invested crypto benchmarks.

The complete engineering path is:

```text
historical data
      ↓
EDA
      ↓
feature engineering
      ↓
model research
      ↓
time-series validation
      ↓
model selection
      ↓
portfolio construction
      ↓
backtesting
      ↓
production inference
      ↓
live market data
      ↓
automated paper trading
      ↓
cloud deployment
      ↓
monitoring
      ↓
public dashboard
```

## Disclaimer

This project is for research, educational, and portfolio demonstration purposes.

The trading environment is simulated. No real-money orders are submitted to Binance or any other exchange.

Nothing in this repository constitutes financial or investment advice.