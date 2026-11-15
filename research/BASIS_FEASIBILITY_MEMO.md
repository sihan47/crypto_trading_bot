# Iteration F: spot/perpetual rich-basis dislocation

2026-11-15. Written before screening code. This is a necessary economic feasibility test, not an execution backtest.

## 1. The Structural Hypothesis
Urgent leveraged demand may pay a premium for perpetual exposure when arbitrage balance sheets are constrained. Buy spot / short equal BTC perpetual to capture convergence. The counterparty accepts immediacy/financing costs; that does not mean their total directional trade must lose. If the observed premium cannot pay friction, discard immediately.

## 2. Feature Engineering & Regime Filter
One core feature: same-hour traded perpetual close / spot close -1. Necessary entry premium >=1.25 times full round-trip friction. Spot fee .001, perp fee .0006, slippage .0005 per leg/side -> complete round trip .0052, buffered threshold .0065 =65bps. This threshold is cost-derived, not fitted. Test hurdle multiplier +/-20% (52/78bps) and doubled friction (130bps). Price-bar basis is NOT executable bid/ask basis.

## 3. Execution & Risk Logic
Only if sufficiently many candidate events: next-hour equal-BTC spot long/perp short, margin funded independently, actual funding flows, no leverage financing assumption, zero-basis convergence or 24h exit and 1% equity stop. That full backtest would use the existing carry engine with separately audited exit semantics and per-leg delay. This screen places no hypothetical orders. Flag candidate hours and separate episodes; require candidates spaced at least24h, consistent with one-day inventory horizon. Never present a screen as portfolio returns.

## 4. Falsification Criteria
Exposed-history holdout2025-2026/11/13 must contain >=30 day-separated events before deeper research. At least30 also required at +/-20% hurdle and double-friction stress to justify further examination. If insufficient, reject this retail-friction setup without computing Sharpe/PF. If sufficient, a subsequent full execution backtest must pass the established Sharpe1.5/PF1.3/DD15%/trade-count/concentration/margin/delay tests. No fee reduction or maker-fill assumption to rescue feasibility.
