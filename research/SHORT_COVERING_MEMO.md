# Iteration E: aggressive-buy short-covering continuation

2026-11-15, written before implementation and backtest. Research only; live bot untouched.

## 1. The Structural Hypothesis
Rising price, declining BTC-denominated OI and aggressive taker buying may identify short-covering pressure. Constrained shorts pay immediacy costs. The hypothesized edge is continuation after the first observed impulse; the opposing mechanism is exhaustion once covering ends. OI does not identify the closing side or prove forced liquidation. This is explicitly an inference, not a liquidation label.

## 2. Feature Engineering & Regime Filter
Three entry features: event-hour price log return / prior-week standard deviation >=2; OI quantity log change / prior-week standard deviation <=-1; latest available 5m taker buy/sell ratio >=2 (two units aggressive buys per unit sells). The ratio is a local confirmation, not a sum/average pretending to represent hourly flow. Day=24h risk horizon, week=168h standardization. One sigma rejects tiny OI churn, two sigma denotes an unusually positive price impulse; both are unoptimized thresholds tested +/-20%.

Metrics retain label+65min availability; event price refers to its corresponding completed hour, not the later delivery hour. Missing, stale or invalid observations suppress signals. Additional delivery delays of 1h/2h are separate falsification scenarios.

## 3. Execution & Risk Logic
Spot long only, next 15m open after availability. Event-hour low is structural invalidation. Planned risk .005 equity, <=1x cash spot, <=.001 previous-bar dollar volume, trailing multiplier2 daily sigma, max24h. Fee .001 and slippage .0005 per side, both doubled in stress. No perpetual position/funding liability. Conservative gap exits, entry-bar stops, no same-bar high hindsight. Economic/risk parameters all independently +/-20%, plus joint corners.

## 4. Falsification Criteria
Development 2023-24, exposed-history temporal holdout 2025-2026/11/13 (not pristine OOS). Require holdout Sharpe>=1.5, PF>=1.3, marked DD<=15%, >=30 round trips; positive net PnL after removing top5% trades; all sensitivity returns >0; doubled cost and extra data delays >0. If continuation fails, reject; do not retrofit a session or a winning perturbation.

Source definition: https://www.cmegroup.com/education/lessons/open-interest . OI describes outstanding contracts, not directional forced flow. Data fields from Binance public metrics/REST, with provenance and caveats in BTC_DERIVATIVES_DATA.md. The continuation mechanism is our testable hypothesis, not a claim that CME or Binance proved this alpha.
