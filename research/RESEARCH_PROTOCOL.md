# Additional spot strategy research protocol

Defined before evaluating the new candidates. This is exploratory research:
the same underlying 2023–2026 history has already been inspected. Splitting
chronologically does not turn these data into a pristine holdout.

Sources motivating families (adaptations, not claimed replications):
- Liu & Tsyvinski, cryptocurrency time-series momentum:
  https://www.nber.org/papers/w24877
- Cryptocurrency moving-average bands and transaction costs (abstract only):
  https://www.tandfonline.com/doi/abs/10.1080/13504851.2024.2394211
- Original channel breakout rules: https://www.turtletrader.com/rules/
- Conventional MACD direction:
  https://www.fidelity.com/learning-center/trading-investing/technical-analysis/technical-indicator-guide/macd

Nine rules, each on 1h and 4h candles: 20/50-day SMA with +/-1% bands;
8/32-day EMA crossover with +/-1% buffer; 28/84-day momentum with +/-1%
thresholds; 20/10 and 55/20-day entry/exit channels; conventional MACD
12/26/9 and MACD with a 20-day SMA entry/trend-exit filter. MACD periods
are bars; other lookbacks are days. No shorting, leverage or averaging down.

No parameter tuning after observing results. Compare all 18 candidates and
cash, current regime_v2 and buy-and-hold benchmarks. Complete 1h/4h candles
only. Full comparison starts 2023-01-01, with 2022 used for indicator warm-up.
Same fee 0.10% and slippage 0.05% per side, full allocation, next-open fills.
These results do not represent GPT review or fixed-lot live execution.

Annual expanding selection: choose ONE shared candidate for BTC and BNB
using earlier years only; evaluate 2024, 2025 and 2026-to-data-end separately.
Score = mean(return_pct + 0.5 * max_drawdown_pct), equally weighting each
asset/year. Cash has score zero and is the fallback. Each candidate must
have at least three round trips in aggregate per asset in the training set.
Save all scores, chosen names and annual test results; double fee/slippage
for the chosen forward-year results without reselecting. Annual test blocks
begin in cash and liquidate at end (not a continuous portfolio simulation).
The selection rule is a simple research convention, not an optimal objective.

Audit: validate OHLC ordering, duplicates and negative volume; compare
standard MACD with the existing inverse direction without editing production.
Test causality and simulator hand-calculations before interpreting results.
Do not modify config, VPS processes or order execution.
