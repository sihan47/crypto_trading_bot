# Forum research and final recommendation protocol

Freeze these rules before inspecting their results. Research only; never
restart or alter the running VPS bot during this work.

## 1. Bollinger/VWAP forum adaptation

Source: https://www.reddit.com/r/algotrading/comments/1lmrjl6/
1h BB(42, 2.5), or six consecutive closes above VWAP plus Wilder RSI(5)>55
and Wilder ADX(5)>45. Exit at +3%, -1.5%, 1075 minutes, or a closed 1h bar
returning inside the upper band. VWAP anchor was unspecified: report UTC
session and rolling 24h versions separately, plus BB-only baseline.

Adapt to the user's spot account: long only, one position, no pyramiding.
This is NOT a replication of the author's long/short three-position returns.
Use 15m candles to resolve stops; if both stop and target touch, stop first.
Gaps through stops fill at the worse open. Signal entry at next 1h open;
1075 minutes rounds up to 1080 minutes on 15m data. Reject an entry at a
missing-data boundary. Mark final position at liquidation cost. Use the
same 0.10% fee + 0.05% slippage per side, separately show the author's
0.025% fee with zero slippage as a sensitivity, never as account economics.

## 2. Multi-asset breakout adaptation

Source: https://www.reddit.com/r/algotrading/comments/1lemuu0/
Original full ranking rules were not published. Our predeclared hypothesis:
fixed six-pair universe BTC, BNB, ETH, SOL, XRP, ADA quoted in USDT;
completed daily 20-day breakout and 10-day exit; positive 30-day return,
close above SMA20, 20-day average quote volume >=10m USDT, channel width
>=1%. Rank qualifying entries by 30-day momentum; at most one asset held.
Hold until 10-day low exit, rather than churn whenever ranks change.
Signals at daily close, trade next open, cash when no valid entry.

Use public spot hourly data for the additional assets; aggregate complete
UTC days only. No future volume ranking, funding, leverage or shorting.
This is a survivor-biased fixed-universe feasibility test, NOT historical
top-60 selection. Do not claim the original forum strategy was reproduced.
Compare to the same rules restricted to BTC/BNB and equal-weight buy/hold.

## Assessment

2023-01-01 through 2026-11-13, with 2022 warm-up. Report continuous full
results and yearly diagnostics, compare the existing ensemble and frozen
50-day/1% band candidate. All history has previously been inspected; no
pristine holdout claim. Prefer robustness, drawdown and operational fit
over the single highest return. Do not optimize the above after results.
Prepare only the final recommended software with Terra 40%; GPT review
itself is NOT included in historical performance. Manual activation later.
