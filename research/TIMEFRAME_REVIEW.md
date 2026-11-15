# Timeframe comparison: 15m to 4h

Approximately September 2022 through November 13, 2026 UTC. Each asset is simulated independently. Same strategy bar-count parameters across timeframes (different elapsed lookbacks); no tuning in this run. Nine deterministic strategies per timeframe, plus buy-and-hold. GPT and historical news are excluded.

Full allocation, long only, continuous capital, next-open fills; fees 0.10% and slippage 0.05% on each side. Each indicator warms up in cash. Complete candles only, missing bars not filled. Annual slices reset to cash with prior indicator history; they do not compound to the full-period result. 2022 and 2026 are partial years.

Historical rankings across 45 candidates per asset are exploratory, not independent validation. Existing ensemble weights were previously chosen using parts of the same history. Returns are cumulative, not annualized. Daily Sharpe uses 365 days and zero risk-free rate. These are not account returns or a test of the complete GPT bot.

## Highest-return strategy at each timeframe

| Timeframe | BTC strategy | BTC return | BTC max drawdown | BNB strategy | BNB return | BNB max drawdown |
| --- | --- | ---: | ---: | --- | ---: | ---: |
| 15m | ema_adx | -85.33% | -87.52% | zscore | -85.41% | -89.51% |
| 30m | ema_adx | -44.22% | -53.42% | zscore | -43.58% | -59.91% |
| 1h | ema_adx | +11.89% | -47.46% | ema_adx | +9.20% | -45.84% |
| 2h | ema_adx | +45.00% | -59.46% | ema_adx | +85.17% | -49.80% |
| 4h | donchian | +163.61% | -44.33% | sma | +95.47% | -59.92% |

## Overall leaders and annual sensitivity

### BTCUSDT: Highest return ? 4h donchian

Return +163.61%; max drawdown -44.33%; daily Sharpe 0.862; round trips 92.

| Year | Return | Max drawdown | Round trips |
| --- | ---: | ---: | ---: |
| 2022 | -12.11% | -19.51% | 8 |
| 2023 | +128.75% | -15.25% | 18 |
| 2024 | +90.69% | -22.30% | 21 |
| 2025 | -17.93% | -26.49% | 25 |
| 2026 | -18.20% | -26.59% | 21 |

### BTCUSDT: Highest daily Sharpe ? 4h donchian

Return +163.61%; max drawdown -44.33%; daily Sharpe 0.862; round trips 92.

| Year | Return | Max drawdown | Round trips |
| --- | ---: | ---: | ---: |
| 2022 | -12.11% | -19.51% | 8 |
| 2023 | +128.75% | -15.25% | 18 |
| 2024 | +90.69% | -22.30% | 21 |
| 2025 | -17.93% | -26.49% | 25 |
| 2026 | -18.20% | -26.59% | 21 |

### BNBUSDT: Highest return ? 4h sma

Return +95.47%; max drawdown -59.92%; daily Sharpe 0.621; round trips 114.

| Year | Return | Max drawdown | Round trips |
| --- | ---: | ---: | ---: |
| 2022 | +17.98% | -12.17% | 7 |
| 2023 | -2.51% | -35.54% | 27 |
| 2024 | +69.71% | -20.96% | 26 |
| 2025 | +13.29% | -40.92% | 27 |
| 2026 | -27.42% | -39.03% | 27 |

### BNBUSDT: Highest daily Sharpe ? 4h sma

Return +95.47%; max drawdown -59.92%; daily Sharpe 0.621; round trips 114.

| Year | Return | Max drawdown | Round trips |
| --- | ---: | ---: | ---: |
| 2022 | +17.98% | -12.17% | 7 |
| 2023 | -2.51% | -35.54% | 27 |
| 2024 | +69.71% | -20.96% | 26 |
| 2025 | +13.29% | -40.92% | 27 |
| 2026 | -27.42% | -39.03% | 27 |

## All results

| Symbol | Timeframe | Strategy | Return | Max drawdown | Daily Sharpe | Round trips |
| --- | --- | --- | ---: | ---: | ---: | ---: |
| BTCUSDT | 15m | sma | -99.60% | -99.61% | -3.396 | 2154 |
| BTCUSDT | 15m | rsi | -97.74% | -97.96% | -2.916 | 1670 |
| BTCUSDT | 15m | macd | -100.00% | -100.00% | -13.563 | 5876 |
| BTCUSDT | 15m | bollinger | -98.47% | -98.52% | -3.081 | 1750 |
| BTCUSDT | 15m | ema_adx | -85.33% | -87.52% | -1.121 | 1070 |
| BTCUSDT | 15m | donchian | -99.17% | -99.20% | -2.993 | 1702 |
| BTCUSDT | 15m | mean_reversion | -98.57% | -98.61% | -4.374 | 1562 |
| BTCUSDT | 15m | zscore | -91.73% | -92.03% | -2.001 | 818 |
| BTCUSDT | 15m | regime_v2 | -93.55% | -94.30% | -2.243 | 1143 |
| BTCUSDT | 15m | buy_hold | +238.79% | -53.84% | 0.854 | 1 |
| BTCUSDT | 30m | sma | -84.38% | -88.12% | -1.070 | 1034 |
| BTCUSDT | 30m | rsi | -85.57% | -86.09% | -1.374 | 837 |
| BTCUSDT | 30m | macd | -99.97% | -99.97% | -6.347 | 2837 |
| BTCUSDT | 30m | bollinger | -87.24% | -87.54% | -1.425 | 860 |
| BTCUSDT | 30m | ema_adx | -44.22% | -53.42% | -0.230 | 520 |
| BTCUSDT | 30m | donchian | -79.93% | -82.37% | -0.888 | 794 |
| BTCUSDT | 30m | mean_reversion | -89.82% | -89.99% | -2.356 | 804 |
| BTCUSDT | 30m | zscore | -60.75% | -64.19% | -0.582 | 402 |
| BTCUSDT | 30m | regime_v2 | -68.21% | -73.21% | -0.904 | 567 |
| BTCUSDT | 30m | buy_hold | +239.26% | -53.74% | 0.854 | 1 |
| BTCUSDT | 1h | sma | -51.76% | -69.48% | -0.324 | 514 |
| BTCUSDT | 1h | rsi | -49.93% | -57.00% | -0.378 | 441 |
| BTCUSDT | 1h | macd | -97.88% | -97.93% | -2.856 | 1390 |
| BTCUSDT | 1h | bollinger | -61.92% | -65.61% | -0.560 | 434 |
| BTCUSDT | 1h | ema_adx | +11.89% | -47.46% | 0.251 | 275 |
| BTCUSDT | 1h | donchian | -55.53% | -65.05% | -0.384 | 399 |
| BTCUSDT | 1h | mean_reversion | -70.69% | -72.67% | -1.163 | 412 |
| BTCUSDT | 1h | zscore | -7.48% | -45.08% | 0.090 | 203 |
| BTCUSDT | 1h | regime_v2 | -7.53% | -63.14% | 0.069 | 305 |
| BTCUSDT | 1h | buy_hold | +239.86% | -53.74% | 0.854 | 1 |
| BTCUSDT | 2h | sma | +37.20% | -56.71% | 0.389 | 245 |
| BTCUSDT | 2h | rsi | -46.41% | -57.62% | -0.310 | 223 |
| BTCUSDT | 2h | macd | -85.54% | -86.21% | -1.323 | 679 |
| BTCUSDT | 2h | bollinger | -0.15% | -53.74% | 0.165 | 222 |
| BTCUSDT | 2h | ema_adx | +45.00% | -59.46% | 0.432 | 134 |
| BTCUSDT | 2h | donchian | +1.82% | -51.53% | 0.184 | 201 |
| BTCUSDT | 2h | mean_reversion | -49.80% | -55.48% | -0.557 | 211 |
| BTCUSDT | 2h | zscore | +8.30% | -37.76% | 0.215 | 95 |
| BTCUSDT | 2h | regime_v2 | +41.44% | -35.44% | 0.450 | 160 |
| BTCUSDT | 2h | buy_hold | +240.85% | -53.52% | 0.854 | 1 |
| BTCUSDT | 4h | sma | +101.49% | -42.07% | 0.678 | 124 |
| BTCUSDT | 4h | rsi | -24.30% | -51.67% | -0.044 | 120 |
| BTCUSDT | 4h | macd | -46.98% | -56.54% | -0.295 | 345 |
| BTCUSDT | 4h | bollinger | +14.74% | -43.19% | 0.265 | 112 |
| BTCUSDT | 4h | ema_adx | +151.65% | -30.63% | 0.837 | 66 |
| BTCUSDT | 4h | donchian | +163.61% | -44.33% | 0.862 | 92 |
| BTCUSDT | 4h | mean_reversion | -1.57% | -29.38% | 0.110 | 111 |
| BTCUSDT | 4h | zscore | +8.04% | -44.83% | 0.215 | 45 |
| BTCUSDT | 4h | regime_v2 | +93.30% | -29.26% | 0.734 | 77 |
| BTCUSDT | 4h | buy_hold | +236.70% | -53.45% | 0.854 | 1 |
| BNBUSDT | 15m | sma | -99.64% | -99.64% | -3.324 | 2041 |
| BNBUSDT | 15m | rsi | -98.45% | -98.82% | -2.654 | 1657 |
| BNBUSDT | 15m | macd | -100.00% | -100.00% | -12.062 | 5952 |
| BNBUSDT | 15m | bollinger | -98.09% | -98.31% | -2.403 | 1665 |
| BNBUSDT | 15m | ema_adx | -95.90% | -96.23% | -1.776 | 1084 |
| BNBUSDT | 15m | donchian | -99.40% | -99.40% | -2.890 | 1659 |
| BNBUSDT | 15m | mean_reversion | -97.35% | -97.47% | -3.548 | 1585 |
| BNBUSDT | 15m | zscore | -85.41% | -89.51% | -1.178 | 825 |
| BNBUSDT | 15m | regime_v2 | -94.91% | -95.45% | -2.245 | 1171 |
| BNBUSDT | 15m | buy_hold | +124.52% | -60.52% | 0.633 | 1 |
| BNBUSDT | 30m | sma | -88.99% | -90.47% | -1.159 | 1031 |
| BNBUSDT | 30m | rsi | -69.39% | -75.16% | -0.596 | 853 |
| BNBUSDT | 30m | macd | -99.98% | -99.98% | -5.607 | 2884 |
| BNBUSDT | 30m | bollinger | -88.92% | -90.78% | -1.277 | 833 |
| BNBUSDT | 30m | ema_adx | -83.14% | -83.47% | -0.933 | 562 |
| BNBUSDT | 30m | donchian | -92.40% | -92.99% | -1.300 | 822 |
| BNBUSDT | 30m | mean_reversion | -78.69% | -82.60% | -1.237 | 801 |
| BNBUSDT | 30m | zscore | -43.58% | -59.91% | -0.195 | 397 |
| BNBUSDT | 30m | regime_v2 | -87.26% | -89.77% | -1.588 | 566 |
| BNBUSDT | 30m | buy_hold | +125.00% | -60.39% | 0.633 | 1 |
| BNBUSDT | 1h | sma | -50.21% | -61.58% | -0.277 | 516 |
| BNBUSDT | 1h | rsi | -62.11% | -70.11% | -0.505 | 430 |
| BNBUSDT | 1h | macd | -98.48% | -98.77% | -2.573 | 1431 |
| BNBUSDT | 1h | bollinger | -34.43% | -51.31% | -0.098 | 422 |
| BNBUSDT | 1h | ema_adx | +9.20% | -45.84% | 0.239 | 275 |
| BNBUSDT | 1h | donchian | -72.30% | -77.12% | -0.593 | 406 |
| BNBUSDT | 1h | mean_reversion | -41.63% | -56.85% | -0.415 | 406 |
| BNBUSDT | 1h | zscore | -26.08% | -42.11% | -0.036 | 190 |
| BNBUSDT | 1h | regime_v2 | -40.75% | -57.54% | -0.271 | 291 |
| BNBUSDT | 1h | buy_hold | +125.73% | -60.39% | 0.633 | 1 |
| BNBUSDT | 2h | sma | +10.74% | -45.99% | 0.244 | 245 |
| BNBUSDT | 2h | rsi | -38.17% | -55.39% | -0.141 | 215 |
| BNBUSDT | 2h | macd | -79.99% | -83.96% | -0.873 | 702 |
| BNBUSDT | 2h | bollinger | -49.54% | -63.09% | -0.260 | 202 |
| BNBUSDT | 2h | ema_adx | +85.17% | -49.80% | 0.585 | 128 |
| BNBUSDT | 2h | donchian | +61.68% | -37.77% | 0.494 | 185 |
| BNBUSDT | 2h | mean_reversion | -33.47% | -58.54% | -0.244 | 196 |
| BNBUSDT | 2h | zscore | +9.14% | -43.84% | 0.232 | 90 |
| BNBUSDT | 2h | regime_v2 | -30.61% | -63.88% | -0.156 | 141 |
| BNBUSDT | 2h | buy_hold | +124.92% | -59.91% | 0.633 | 1 |
| BNBUSDT | 4h | sma | +95.47% | -59.92% | 0.621 | 114 |
| BNBUSDT | 4h | rsi | +18.98% | -41.63% | 0.295 | 105 |
| BNBUSDT | 4h | macd | -27.43% | -54.78% | -0.012 | 381 |
| BNBUSDT | 4h | bollinger | +16.79% | -47.86% | 0.287 | 103 |
| BNBUSDT | 4h | ema_adx | +13.46% | -49.12% | 0.271 | 65 |
| BNBUSDT | 4h | donchian | +9.20% | -56.00% | 0.238 | 94 |
| BNBUSDT | 4h | mean_reversion | +31.12% | -32.37% | 0.372 | 107 |
| BNBUSDT | 4h | zscore | -5.00% | -45.27% | 0.114 | 44 |
| BNBUSDT | 4h | regime_v2 | +29.08% | -39.44% | 0.353 | 64 |
| BNBUSDT | 4h | buy_hold | +126.46% | -59.62% | 0.633 | 1 |

Reproduce: `python research/compare_timeframes.py`, then `python research/summarize_timeframes.py`. No live configuration or VPS changes.
