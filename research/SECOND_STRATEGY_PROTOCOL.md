# Second independent strategy candidate

User requested two strategies rather than one. Keep band50 frozen. Before
testing a second family, define these four additional candidates:

1. RSI2 pullback on 4h candles: close above 100-day SMA, RSI(2) crosses
   below 15; sell when RSI2 > 30 or close below the trend SMA.
2. Same rules on daily candles.
3. Volatility-normalized trend on 4h: (close - 65-day EMA) / 65-day std;
   enter above +0.5 and exit below -0.5; preserve position in between.
4. Same volatility-normalized rules on daily candles.

RSI source: https://alvarezquanttrading.com/wp-content/uploads/2019/02/StrategyDescriptionResults.pdf
This is a MARKET-entry crypto adaptation; the author's equity version uses
a next-day limit at close minus 0.5 ATR10. We do not assume limit fills.
Its stock universe, ranking and 10-position sizing are not reproduced.

Z-score discussion: https://www.reddit.com/r/algotrading/comments/1qgcq2h/
The author did not disclose thresholds. +/-0.5 are our predeclared choices;
no attempt to reproduce their advertised CAGR. This is trend-following,
not a complementary mean-reversion strategy merely because it uses z-score.

Data/costs: same 2023-01-01 to 2026-11-13 data, prior warm-up, full allocation,
long only, next open, fees 0.1% + slippage 0.05% each side. Report yearly,
double-cost and both BTC/BNB results. All tests are exploratory on already
seen history. Choose a distinct second rule only if evidence is acceptable;
otherwise present it as research-only, not a forced profitable recommendation.
