# Existing strategies on 4-hour candles: full-history comparison

Continuous cached history, approximately September 2022 through November 13, 2026 UTC. Exact ranges and parameters are recorded in `existing_4h_full_evaluation.json`.

Each strategy uses its existing dataclass defaults. The regime_v2 ensemble uses the current local config parameters (trend/breakout/reversion weights 0.4/0.3/0.3); GPT is excluded. No parameters were optimized in this run. The ensemble weights were previously selected using portions of this history, so these are not blind validation results.

Execution: single long position with full allocation, signal at candle close and fill at the next available consecutive candle open. Fee 0.10% and slippage 0.05% on each side. Initial cash 10,000 USDT independently per symbol; no annual resets; final positions liquidated. Results do not reproduce live fixed-lot orders, GPT/news review or polling delays.

Only groups containing all sixteen 15-minute candles become 4-hour candles. Each asset has one retained time gap. Indicator warm-up is time in cash. Buy-and-hold enters on the second candle open, so its entry price differs slightly from the earlier 15-minute benchmark.

| Strategy | BTC return | BTC max drawdown | BTC round trips | BNB return | BNB max drawdown | BNB round trips |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| sma | +101.49% | -42.07% | 124 | +95.47% | -59.92% | 114 |
| rsi | -24.30% | -51.67% | 120 | +18.98% | -41.63% | 105 |
| macd | -46.98% | -56.54% | 345 | -27.43% | -54.78% | 381 |
| bollinger | +14.74% | -43.19% | 112 | +16.79% | -47.86% | 103 |
| ema_adx | +151.65% | -30.63% | 66 | +13.46% | -49.12% | 65 |
| donchian | +163.61% | -44.33% | 92 | +9.20% | -56.00% | 94 |
| mean_reversion | -1.57% | -29.38% | 111 | +31.12% | -32.37% | 107 |
| zscore | +8.04% | -44.83% | 45 | -5.00% | -45.27% | 44 |
| regime_v2 | +93.30% | -29.26% | 77 | +29.08% | -39.44% | 64 |
| buy_hold | +236.70% | -53.45% | 1 | +126.46% | -59.62% | 1 |

## What 4h means

Six candles per UTC day: 00:00?04:00, 04:00?08:00, and so on. Each contains its first price, highest price, lowest price, last price and summed volume. A candle is usable only once closed. A 20-bar average on 4h data spans 80 hours, not 20 hours. The current ensemble fast/slow windows 48/192 correspond to 8/32 days. This timeframe is neither a required trading frequency nor a fixed holding period.

ATR is a risk filter, not an independently traded strategy. GPT, sentiment and news are not simulated. The table evaluates individual existing rules plus the current ensemble technical component, not the historical legacy GPT voting system.

Reproduce: `python research/evaluate_existing_4h.py`. No production settings or VPS state were changed.
