# Iteration D: crowded-long deleveraging proxy

Published before implementation/backtesting, 2026-11-15. Same exposed-history development/holdout partition as A/B/C. This is NOT a pristine OOS experiment and is NOT an actual liquidation classifier.

## 1. The Structural Hypothesis
Crowded leveraged longs may pay immediacy costs when forced to close during a price decline. An unusually large decline in BTC-denominated OI concurrent with price weakness is a more direct deleveraging proxy than volume alone. After selling pressure is exhausted, spot liquidity providers may earn reversal. OI also falls on voluntary closures and does not identify direction of forced trades; permanent news repricing remains a failure mode.

## 2. Feature Engineering & Regime Filter
Exactly three entry features: event-hour log price change divided by prior-week hourly-return standard deviation <= -2; event-hour log OI-quantity change divided by its prior-week standard deviation <= -2; PRE-event global account long/short ratio above its previous-week 75th percentile. Use OI quantity, never OI USDT value as a substitute. Account ratios measure accounts, not notional exposure. Price event uses the closed hour corresponding to the metric snapshot interval, not the later delivery hour. Only contiguous hourly metric observations contribute OI changes. Missing/invalid fields suppress entry. Availability = raw metric label +65min; compare additional +1h/+2h delays.

## 3. Execution & Risk Logic
Spot long after data availability, next consecutive 15m open; event-hour low invalidation; two-daily-sigma trailing stop, maximum 24h; .5% planned equity risk, <=1x spot notional, <=.1% previous 15m dollar volume. Same A/B cost model: .1% fee +.05% slippage each side, doubled stress. No futures position, zero funding. Actual funding, basis and other ratios are included in the data panel for later separately specified hypotheses, not added as extra knobs here.

## 4. Falsification Criteria
Holdout 2025 through 2026-11-13; development 2023-24. Sharpe >=1.5, PF>=1.3, drawdown<=15%, >=30 trades, positive after removing top5% trades; every +/-20% parameter perturbation and common corner positive; doubled cost positive. Day=24 risk horizon, week=168 standardization/crowding window, shock=2, crowding tail=.25, trail=2, max_hours=24, risk=.005, liquidity_cap=.001. Additional 1h and 2h metric-delivery delays must retain positive returns to qualify. No post-result session/threshold selection. Reject if delayed data remove the signal, rather than using future snapshots to repair it.
