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

### Continuous full-history run

The full cached interval was also simulated continuously, without resetting
capital each year: BTC starts 2022-08-31 16:00 UTC; BNB starts 2022-09-01
00:00 UTC. Both end with the 2026-11-13 23:45 UTC opening candle.
BTC has 147,386 bars and BNB 147,354; each series contains one gap.
Indicator warm-up is included as time in cash. Buy-and-hold starts immediately.
The same costs and full-allocation sizing apply; GPT remains excluded.

| Variant | BTC return | BNB return |
| --- | ---: | ---: |
| Pullback only | -92.85% | -97.10% |
| Squeeze only | -50.46% | -56.07% |
| Combined | -96.33% | -98.76% |
| Buy and hold | +238.79% | +124.52% |

Combined BTC: 1,052 round trips, 12.36% win rate, -96.34% maximum drawdown.
Combined BNB: 1,522 round trips, 18.27% win rate, -98.78% maximum drawdown.
These results describe this implementation and its sizing assumptions, not
the user's account returns or all possible pullback/breakout strategies.
Machine-readable metrics are under each symbol's `periods.full` key.

### Commands

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
