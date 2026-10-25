# Crypto Volatility Forecasting & Paper Trading System

An end-to-end quantitative finance and machine learning project for forecasting cryptocurrency volatility and using those forecasts for risk-aware portfolio allocation, backtesting, production inference, and live paper trading.

## Project Goals

This project aims to:

- collect and validate cryptocurrency market data;
- explore return and volatility behaviour;
- engineer interpretable volatility features;
- compare econometric, machine-learning, and deep-learning models;
- use leakage-safe time-series validation;
- select the strongest model for each asset;
- convert volatility forecasts into portfolio weights;
- control portfolio risk and concentration;
- backtest with realistic trading costs and execution rules;
- retrain frozen model specifications for production;
- generate production portfolio decisions without retraining;
- connect to live Binance market data;
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

## Feature Set

A research pool of 28 candidate features was created.

After a training-only feature audit, the main ML feature set was reduced to seven interpretable predictors:

- current log Garman-Klass volatility;
- 7-day average volatility;
- 30-day average volatility;
- absolute return;
- log volume;
- cross-asset mean log volatility;
- downside return.

The full feature set is retained for research comparisons.

## Leakage Prevention

Time-series leakage is treated as a hard constraint.

The modelling pipeline uses:

- chronological train / validation / test splits;
- no random shuffling of time-series partitions;
- one-row purges at split boundaries;
- training-only feature scaling;
- training-only LSTM target scaling;
- expanding-window validation;
- features available only at forecast time;
- out-of-sample development forecasts for portfolio-rule research;
- a separate final test period.

Purged rows are excluded as supervised targets when their `t + 1` label crosses a split boundary.

Observed information from those rows may still be used as legitimate historical context for later forecasts.

LSTM training sequences may be shuffled during optimization because chronology is preserved inside every sequence.

## Model Research

### Naive Baseline

The 7-day rolling mean was the strongest naive benchmark.

| Asset | Validation RMSE |
| --- | ---: |
| BTC | 0.5086 |
| ETH | 0.4865 |
| SOL | 0.4204 |
| BNB | 0.4829 |

### HAR-GK

HAR-GK uses current, 7-day, and 30-day log Garman-Klass volatility.

| Asset | HAR-GK RMSE |
| --- | ---: |
| BTC | 0.4904 |
| ETH | 0.4553 |
| SOL | 0.3968 |
| BNB | 0.4538 |

HAR-GK improved on the naive benchmark for every asset.

The 7-day volatility component was the strongest HAR coefficient across all four assets.

### EGARCH

An EGARCH(1,1,1) model with Student-t innovations was evaluated using expanding forecasts.

| Asset | EGARCH RMSE |
| --- | ---: |
| BTC | 0.6354 |
| ETH | 0.5283 |
| SOL | 0.4859 |
| BNB | 0.5656 |

EGARCH underperformed the stronger range-based volatility models.

### Ridge and Lasso

Regularized linear models used the compact seven-feature set with training-only scaling.

| Asset | Ridge RMSE | Lasso RMSE |
| --- | ---: | ---: |
| BTC | 0.4886 | 0.4899 |
| ETH | 0.4470 | 0.4468 |
| SOL | 0.3941 | 0.3943 |
| BNB | 0.4512 | 0.4497 |

Ridge and Lasso were strong and stable benchmarks.

### Nonlinear Tree Models

Random Forest and Gradient Boosting were evaluated using training-only time-series hyperparameter tuning.

| Asset | Random Forest | Gradient Boosting |
| --- | ---: | ---: |
| BTC | 0.4865 | 0.4974 |
| ETH | 0.4656 | 0.4499 |
| SOL | 0.3976 | 0.3942 |
| BNB | 0.4630 | 0.4646 |

Tree models did not consistently outperform the simpler regularized models.

### LSTM

The LSTM pipeline was improved with:

- training-only feature scaling;
- training-only target scaling;
- sequence-length tuning;
- hidden-size tuning;
- learning-rate tuning;
- early stopping;
- evaluation across three fixed random seeds.

Initial tuned results:

| Asset | Sequence Length | Hidden Size | Mean RMSE |
| --- | ---: | ---: | ---: |
| BTC | 30 | 16 | 0.4874 |
| ETH | 7 | 32 | **0.4335** |
| SOL | 7 | 32 | 0.3947 |
| BNB | 7 | 32 | 0.4685 |

ETH showed the clearest benefit from sequence modelling.

### VAR and Cross-Asset Volatility

A Vector Autoregression model was used to model BTC, ETH, SOL, and BNB volatility jointly.

A 7-day lag structure was selected using training-period information only.

Initial validation results:

| Asset | VAR RMSE |
| --- | ---: |
| BTC | **0.4612** |
| ETH | 0.4444 |
| SOL | **0.3864** |
| BNB | 0.4507 |

VAR produced the strongest initial results for BTC and SOL.

Training-period Granger-causality tests also showed strong predictive relationships from BTC volatility to ETH, SOL, and BNB volatility.

These are interpreted as predictive relationships rather than proof of economic causation.

## Robust Model Validation

Final model selection used four expanding historical validation windows.

Validation design:

- Initial training window: 1,200 observations
- Validation window: 120 observations
- Step size: 120 observations
- Boundary purge: 1 observation
- Validation folds: 4
- Total robust validation observations: 480 per asset
- Final test start: 2025-11-28

Hyperparameters selected during model research were frozen before robust validation.

### Robust Results

| Asset | Selected Candidate | Robust Mean RMSE |
| --- | --- | ---: |
| BTC | VAR | **0.4471** |
| ETH | LSTM | **0.4231** |
| SOL | VAR | **0.3889** |
| BNB | Ridge | **0.4482** |

## Selected Forecast Models

The final champion models are:

| Asset | Selected Model | Frozen Configuration |
| --- | --- | --- |
| BTC | **VAR** | lag = 7 |
| ETH | **LSTM** | seq = 7, hidden = 32, lr = 0.0005 |
| SOL | **VAR** | lag = 7 |
| BNB | **Ridge** | alpha = 100 |

VAR is trained jointly on all four volatility series, while its BTC and SOL equations are used for portfolio forecasts.

ETH uses the specialist LSTM forecast.

BNB uses Ridge because the simpler regularized model remained strongest.

## Final Forecast Test

The frozen champion models were evaluated once on the final test period.

Test period:

- Start: 2025-11-28
- End: 2026-10-05
- Observations: 312 per asset

| Asset | Model | MAE | RMSE |
| --- | --- | ---: | ---: |
| BTC | VAR | 0.3506 | **0.4483** |
| ETH | LSTM | 0.3566 | **0.4597** |
| SOL | VAR | 0.3402 | **0.4413** |
| BNB | Ridge | 0.3369 | **0.4250** |

Forecast-model choices and hyperparameters were not changed after observing the final test.

## Portfolio Construction

Forecasts are converted into allocations through:

```text
next-day log-volatility forecasts
        ↓
convert to volatility
        ↓
inverse-volatility allocation
        ↓
40% maximum asset weight
        ↓
30-day historical return correlation
        ↓
forecast covariance matrix
        ↓
portfolio volatility estimate
        ↓
20% annual volatility target
        ↓
crypto exposure + cash
        ↓
execution threshold
```

### Why the Strategy Is Long-Only

The forecasting models estimate **volatility**, not expected returns.

They answer:

> How risky is each asset likely to be tomorrow?

They do not answer:

> Will this asset increase or decrease in price tomorrow?

For that reason, high forecast volatility is interpreted as a reason to **reduce position size**, not as evidence that an asset should be shorted.

The portfolio therefore remains long-only and uses cash as the defensive allocation.

A long/short strategy would require an additional expected-return or directional forecasting layer, together with different assumptions for borrowing, funding costs, liquidation risk, and execution.

Keeping the strategy long-only therefore keeps the portfolio construction aligned with what the models actually predict.

## Risk Controls

The portfolio uses:

- 20% annualized volatility target;
- 40% maximum weight per crypto asset;
- 30-day return-correlation lookback;
- maximum risky exposure of 100%;
- no leverage;
- remaining capital held in cash.

## Portfolio Execution

Trading decisions are made by comparing the target portfolio with the current portfolio:

```text
trade weight = target weight - current weight
```

Therefore:

- positive trade weight → buy;
- negative trade weight → sell;
- zero trade weight → no trade.

The strategy does not short assets.

Signals are evaluated daily, but trades are executed only when proposed one-way turnover is at least 5%.

## Position Sizing

Portfolio weights are expressed as fractions of total portfolio equity.

For example:

```text
target BTC weight = 0.18
```

means approximately 18% of portfolio value should be allocated to BTC.

Dollar position size is:

```text
target position value = portfolio equity × target weight
```

Asset quantity is:

```text
target quantity = target position value / current market price
```

The allocation is therefore scale-independent in percentage terms.

However, the current backtest does not model market impact or strategy capacity. A portfolio with $100 million of capital cannot be assumed to obtain the same execution quality as a much smaller portfolio without additional liquidity and slippage analysis.

## Portfolio-Rule Development

Trading rules were evaluated using **480 out-of-sample development forecasts** from the same expanding validation folds used during robust model evaluation.

The predefined rule grid tested:

- rebalance every 1, 3, or 7 days;
- minimum one-way turnover threshold of 0%, 2%, or 5%.

Forecast models remained frozen.

### Development Results

| Rule | Sharpe | Max Drawdown | Avg. Daily Turnover | Transaction Cost | Rebalances |
| --- | ---: | ---: | ---: | ---: | ---: |
| Daily, 0% threshold | **0.8551** | **-19.81%** | 7.18% | 3.45% | 480 |
| Daily, 5% threshold | 0.8545 | -20.09% | **6.15%** | **2.95%** | **293** |

The 5% threshold preserved almost all risk-adjusted performance while materially reducing trading.

The frozen execution rule is therefore:

- evaluate signals daily;
- rebalance only when proposed one-way turnover is at least 5%.

## Backtesting

The strategy backtest uses:

- correct `t → t+1` forecast and return alignment;
- simple returns for portfolio P&L;
- natural portfolio-weight drift;
- cash allocation;
- 10 bps transaction costs;
- 5% minimum turnover threshold;
- no leverage.

### Final Strategy Results

| Metric | Result |
| --- | ---: |
| Total return | **0.74%** |
| Annualized return | **0.87%** |
| Annualized volatility | **23.07%** |
| Sharpe ratio | **0.152** |
| Sortino ratio | **0.228** |
| Calmar ratio | **0.035** |
| Maximum drawdown | **-25.01%** |
| Max drawdown duration | **264 days** |
| Hit rate | **48.87%** |
| 95% historical VaR | **1.56%** |
| 95% Expected Shortfall | **2.49%** |
| Average daily turnover | **7.38%** |
| Total transaction cost | **2.29%** |
| Average risky exposure | **53.39%** |
| Average cash weight | **46.61%** |
| Rebalances | **206** |

The strategy produced modest returns but materially reduced realized risk relative to fully invested crypto benchmarks.

## Benchmark Comparison

| Strategy | Total Return | Annualized Volatility | Sharpe | Max Drawdown |
| --- | ---: | ---: | ---: | ---: |
| BTC Buy & Hold | -5.73% | 44.74% | 0.068 | -39.53% |
| Equal Weight Buy & Hold | -10.00% | 49.76% | 0.000 | -46.92% |
| Equal Weight Daily Rebalanced | -8.67% | 50.26% | 0.039 | -47.20% |
| Volatility Strategy | **+0.74%** | **23.07%** | **0.152** | **-25.01%** |

The strategy's main value is **risk reduction and capital preservation**, rather than aggressive directional return generation.

## Production ML

After research and model selection were complete, the frozen champion specifications were retrained on all available historical data.

Production training data:

- Rows per asset: 2,074
- Start: 2021-01-31
- End: 2026-10-05

Production models:

| Asset | Model | Configuration |
| --- | --- | --- |
| BTC | VAR | lag = 7 |
| ETH | LSTM | seq = 7, hidden = 32, lr = 0.0005, epochs = 52 |
| SOL | VAR | lag = 7 |
| BNB | Ridge | alpha = 100 |

Production artifacts include:

- joint VAR model;
- VAR history state;
- BNB Ridge model;
- BNB feature scaler;
- ETH LSTM checkpoint;
- ETH feature scaler;
- ETH target scaler;
- production manifest.

A separate inference process successfully reloads the saved artifacts and generates forecasts without retraining.

### Production Forecast Smoke Test

Latest historical forecast origin:

```text
2026-10-05
```

Next-day log-volatility forecasts:

| Asset | Forecast |
| --- | ---: |
| BTC | -4.1608 |
| ETH | -3.9151 |
| SOL | -3.8754 |
| BNB | -4.0352 |

### Production Portfolio Decision

The forecasts were passed through the same tested allocation and risk-control pipeline used during research.

Result:

| Asset | Target Weight |
| --- | ---: |
| BTC | 18.32% |
| ETH | 14.33% |
| SOL | 13.77% |
| BNB | 16.16% |
| Cash | 37.42% |

Additional outputs:

- forecast portfolio volatility: 1.67% daily;
- risky exposure: 62.58%;
- cash weight: 37.42%;
- turnover from an all-cash portfolio: 62.58%;
- rebalance decision: yes.

The production decision respects the 20% annual volatility target, concentration controls, maximum risky exposure, and 5% execution threshold.

## Current Findings

1. Cryptocurrency volatility is strongly persistent.
2. The 7-day horizon is the most consistent forecasting signal.
3. HAR-GK is a strong interpretable benchmark.
4. EGARCH is weaker for the selected range-based volatility target.
5. Compact feature sets perform as well as or better than larger feature sets.
6. Ridge and Lasso are strong and stable ML baselines.
7. Tree models add limited incremental value.
8. ETH benefits from sequence modelling.
9. Cross-asset volatility information materially improves BTC and SOL forecasting.
10. BTC appears to be an important volatility transmitter in the multivariate system.
11. Different assets benefit from different forecasting models.
12. No single model family dominates every asset.
13. Final-period forecast performance shows regime sensitivity.
14. Volatility forecasts can be translated into materially lower portfolio risk.
15. The 5% turnover threshold reduces trading without materially reducing Sharpe.
16. Portfolio risk management adds more value here than directional return prediction.
17. Long-only allocation is consistent with the information content of volatility forecasts.
18. Production models can be loaded and used without retraining.
19. The research pipeline now connects directly to executable portfolio targets.

## Project Status

**Current phase: Phase 11 — Live Binance Data**

Completed:

- ✅ Project setup
- ✅ Historical Binance data pipeline
- ✅ Exploratory data analysis
- ✅ Feature engineering
- ✅ Model-ready datasets
- ✅ Leakage-safe time-series splitting
- ✅ Naive volatility baselines
- ✅ HAR-GK
- ✅ EGARCH
- ✅ Feature selection
- ✅ Ridge regression
- ✅ Lasso regression
- ✅ Random Forest
- ✅ Gradient Boosting
- ✅ LSTM modelling and training audit
- ✅ VAR / cross-asset volatility analysis
- ✅ Training-only hyperparameter tuning
- ✅ Robust expanding-window validation
- ✅ Champion model selection
- ✅ Final forecast evaluation
- ✅ Portfolio allocation
- ✅ Maximum-weight risk controls
- ✅ Volatility targeting
- ✅ Cash allocation
- ✅ Forecast covariance estimation
- ✅ Transaction-cost modelling
- ✅ Development-period execution-rule validation
- ✅ Weight-drift-aware backtesting
- ✅ Advanced backtest metrics
- ✅ Historical VaR and Expected Shortfall
- ✅ Drawdown-duration analysis
- ✅ Final benchmark comparison
- ✅ Equity-curve reporting
- ✅ Drawdown reporting
- ✅ Dynamic exposure reporting
- ✅ Turnover reporting
- ✅ Portfolio-weight reporting
- ✅ Production model retraining
- ✅ Production artifact persistence
- ✅ Production model loading
- ✅ Production inference smoke test
- ✅ Forecast-to-portfolio decision pipeline

Current production strategy:

- BTC → VAR, lag 7
- ETH → LSTM, sequence 7, hidden size 32
- SOL → VAR, lag 7
- BNB → Ridge, alpha 100
- 20% annual volatility target
- 40% maximum asset weight
- 30-day correlation lookback
- 5% minimum turnover threshold
- 10 bps transaction-cost assumption
- maximum risky exposure of 100%
- long-only
- no leverage

Next:

- 🔄 Live Binance closed-candle ingestion
- ⬜ Live feature generation
- ⬜ Live production inference
- ⬜ Portfolio-state persistence
- ⬜ Paper broker
- ⬜ Paper trading loop
- ⬜ Production hardening
- ⬜ Docker and CI/CD
- ⬜ Deployment
- ⬜ Monitoring and observability

## Model Families Evaluated

- naive volatility baselines;
- HAR-GK;
- EGARCH;
- Ridge;
- Lasso;
- Random Forest;
- Gradient Boosting;
- LSTM;
- VAR / cross-asset volatility modelling.

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
```

Research notebooks are kept as experiment records, while reusable and tested logic is moved into `src/`.