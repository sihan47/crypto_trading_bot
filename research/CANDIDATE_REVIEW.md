# Additional strategy research: findings

## Conclusion

The 4h 50-day SMA with a +/-1% band improves historical total return over the existing ensemble on both assets, but has larger drawdowns. It is a candidate for further engineering and validation, not an approved live replacement. BTC shows better local parameter stability than BNB. No GPT decisions or actual fixed-lot orders were simulated.

Comparison interval: 2023-01-01 to 2026-11-13 UTC. The 2022 data are used for warm-up. This deliberately differs from earlier full-cache comparisons; all rows below use the same new start. Same full-allocation next-open simulator, 0.10% fee plus 0.05% slippage per side. Results are cumulative, not annualized.

## Sources and implementation audit

- Cryptocurrency time-series momentum is supported as a research hypothesis by [Liu and Tsyvinski](https://www.nber.org/papers/w24877); their finding is not a guarantee for this sample or these exact parameters.
- The [moving-average band paper](https://www.tandfonline.com/doi/abs/10.1080/13504851.2024.2394211) motivated a no-trade buffer; only its abstract was accessible. Our 1% band is a research choice, not their claimed optimum.
- [Turtle rules](https://www.turtletrader.com/rules/) motivated day-based channels; our long-only fixed-allocation version is not the full Turtle risk/position-sizing system.
- The repo MACD buys bearish crosses and sells bullish crosses, the opposite of [conventional MACD](https://www.fidelity.com/learning-center/trading-investing/technical-analysis/technical-indicator-guide/macd). New standard-direction candidates were tested without changing production. Standard MACD alone still performs poorly; reversing the signals is not a sufficient improvement.
- The [2020 trend-following paper](https://arxiv.org/html/2009.12155v1) assumes negligible fees, spread and slippage. Its headline return must not be compared directly to our cost-inclusive results.
- Cached data checks found no duplicated timestamps, inconsistent OHLC ordering or negative volume. Each asset has a time gap, not interpolated. This is a structural audit, not independent verification of every candle against the exchange.

## Selected candidate versus benchmarks

| Asset | Strategy | Return | Max drawdown | Round trips |
| --- | --- | ---: | ---: | ---: |
| BTCUSDT | 4h:sma50_band | +313.60% | -35.77% | 33 |
| BTCUSDT | 4h:current_ensemble | +106.67% | -29.26% | 73 |
| BTCUSDT | 4h:buy_hold | +311.96% | -53.45% | 1 |
| BNBUSDT | 4h:sma50_band | +113.96% | -43.76% | 46 |
| BNBUSDT | 4h:current_ensemble | +53.33% | -34.80% | 60 |
| BNBUSDT | 4h:buy_hold | +156.19% | -59.62% | 1 |

Rules: on a completed 4h candle, buy above 1.01 times the trailing 300-bar SMA; sell below 0.99 times that SMA; between the two, retain the current state. One position only. Signals execute at the following open. No leverage, shorting or pyramiding. This produces fewer trades, not more frequent trading.

## Chronological expanding selection

Nine frozen rules on 1h/4h (18 candidates). Earlier years select one shared rule for both assets using mean return plus half the negative maximum drawdown. Cash fallback and minimum training trade count are defined in RESEARCH_PROTOCOL.md. Each next year starts in cash. These are separate annual evaluations, not a stitched continuous portfolio. All three selections chose 4h:sma50_band.

| Test year | Asset | Selected return | Doubled-cost return | Existing ensemble | Buy and hold |
| --- | --- | ---: | ---: | ---: | ---: |
| 2024 | BTCUSDT | +64.77% | +59.90% | +53.18% | +120.40% |
| 2024 | BNBUSDT | +49.82% | +43.23% | +43.50% | +126.45% |
| 2025 | BTCUSDT | +0.69% | -2.28% | +3.70% | -6.88% |
| 2025 | BNBUSDT | +24.35% | +19.96% | +10.42% | +22.13% |
| 2026 | BTCUSDT | +1.23% | -1.76% | -0.42% | -22.47% |
| 2026 | BNBUSDT | -4.70% | -7.52% | -2.91% | -27.70% |

2026 ends November 13. Previously inspected history and prior ensemble selection prevent calling these pristine out-of-sample results. In 2026 BNB still loses and underperforms the existing ensemble. Doubled costs make BTC 2025 and 2026 negative. No universal improvement is established.

## Post-selection sensitivity (not retuning)

Keep the selected 50-day/1% rule unchanged. Neighbouring 40/60-day and 0.5/1.5% settings were diagnostic only.

| Asset | Setting | Return | Doubled-cost return |
| --- | --- | ---: | ---: |
| BTCUSDT | 50d_0.01 | +313.60% | +274.61% |
| BTCUSDT | 40d_0.005 | +292.53% | +234.82% |
| BTCUSDT | 40d_0.015 | +348.66% | +313.75% |
| BTCUSDT | 60d_0.005 | +281.51% | +237.36% |
| BTCUSDT | 60d_0.015 | +323.26% | +295.04% |
| BNBUSDT | 50d_0.01 | +113.96% | +86.38% |
| BNBUSDT | 40d_0.005 | +25.13% | -1.27% |
| BNBUSDT | 40d_0.015 | +44.65% | +27.53% |
| BNBUSDT | 60d_0.005 | +147.36% | +111.63% |
| BNBUSDT | 60d_0.015 | +171.90% | +152.25% |

BTC remains positive across these neighbours; BNB is materially more sensitive and one neighbour loses at doubled costs. Do not replace the frozen setting with the best-looking neighbour.

## All candidate returns

| Candidate | BTC return | BNB return |
| --- | ---: | ---: |
| 1h:sma20_band | +222.62% | +81.75% |
| 1h:sma50_band | +283.87% | +92.23% |
| 1h:ema8_32_band | +142.23% | +137.70% |
| 1h:momentum28 | +148.82% | -7.03% |
| 1h:momentum84 | +186.47% | +95.62% |
| 1h:channel20_10 | +157.95% | +33.65% |
| 1h:channel55_20 | +68.55% | +42.71% |
| 1h:macd_standard | -94.20% | -95.31% |
| 1h:macd_trend_filter | -70.17% | -79.98% |
| 1h:legacy_inverse_macd | -96.86% | -97.96% |
| 4h:sma20_band | +199.74% | +77.21% |
| 4h:sma50_band | +313.60% | +113.96% |
| 4h:ema8_32_band | +157.63% | +144.32% |
| 4h:momentum28 | +151.39% | +9.56% |
| 4h:momentum84 | +205.28% | +106.87% |
| 4h:channel20_10 | +184.16% | +7.82% |
| 4h:channel55_20 | +66.38% | +82.54% |
| 4h:macd_standard | -1.54% | -65.93% |
| 4h:macd_trend_filter | +27.79% | -18.63% |
| 4h:legacy_inverse_macd | -37.92% | -10.20% |
| 4h:current_ensemble | +106.67% | +53.33% |
| 4h:buy_hold | +311.96% | +156.19% |

## Reproduce and scope

1. `python -m unittest discover -s tests -p test_research_candidates.py` (four passing tests: causality, warm-up, MACD direction, hand-calculated execution costs).
2. `python research/evaluate_candidates.py`
3. `python research/check_candidate_sensitivity.py`

Machine-readable files preserve all candidate metrics, annual selection scores and sensitivity checks with input SHA-256 hashes. Scripts are research-only. No config changes, production strategy edits, VPS deploys or trading state changes were performed. No API account calls were made.
