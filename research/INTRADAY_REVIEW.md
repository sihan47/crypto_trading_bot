# 15-minute pullback and squeeze candidate — rejected for live use

This candidate increases trade frequency but fails the historical cost test.
It is research code only, not wired into the production runner or config.
Do not treat uploading this directory as activating a trading strategy.

## Rules fixed before this comparison

Only completed 4-hour candles supply the direction filter: SMA20 > SMA50
and the 4-hour close above SMA50. All entries/exits use closed 15-minute
candles, with simulated execution at the following open.

- Pullback: recent RSI2 below 10, price starts recovering but remains below
  its 20-bar mean, and distance to the mean exceeds 1.5 times estimated
  round-trip costs. Exit on recovery (mean or RSI2 >= 70), or after 8 bars.
- Squeeze: Bollinger bandwidth recently in its trailing 192-bar bottom
  decile; break above the previous upper band and 20-bar high, with volume
  above 1.2 times its previous 20-bar mean. Exit on failed breakout, a
  3-ATR trailing close, or after 24 bars.
- Both: 2-ATR close-based stop, exit when higher-timeframe trend breaks,
  4-bar cooldown, single long position. These are not exchange stop orders.

The rules are adaptations, not published claims of profitability on crypto:
[John Bollinger's rules](https://www.bollingerbands.com/bollinger-band-rules)
and [Connors' pullback guide](https://www.tradingmarkets.com/media/2012/ConnorsRSI-Pullbacks-Guidebook.pdf).

## Results

2026-05-02 through 2026-11-13 UTC; fee 0.10% plus slippage 0.05% per side.
Returns use a fully allocated simulated position, not the bot's fixed lots.

| Symbol | Pullback only | Squeeze only | Combined | Combined round trips |
| --- | ---: | ---: | ---: | ---: |
| BTCUSDT | -20.46% | -6.55% | -25.67% | 97 |
| BNBUSDT | -30.75% | -12.25% | -39.23% | 164 |

Every variant also loses in each of 2023, 2024 and 2025 after costs.
Full metrics and doubled-cost stress results are in `intraday_evaluation.json`.
These periods were inspected during earlier research; this is exploratory,
not an untouched out-of-sample test. No GPT decisions were replayed. A 40%
Terra review cannot be assumed to repair these losses without evidence.

Six offline tests cover completed higher candles, future-data independence,
next-open stop reference, time exits/cooldowns, position permissions and cost
filters. Passing software tests does not establish a profitable strategy.

## Reproduce

Use the existing cached public candles, or prepare them first:

```text
python research/prepare_ensemble_data.py --end 2026-11-14
python -m unittest discover -s tests -p test_intraday.py
python research/evaluate_intraday.py
```

This change deliberately leaves runtime, account credentials, existing
positions and production configuration untouched. Before any future runtime
integration, actual fill/state tracking must replace research virtual-position
assumptions, and trading costs must be retested for the intended account.
