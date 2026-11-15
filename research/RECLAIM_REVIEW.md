# G review: recovery confirmation does not establish an edge

## What changed structurally

Prior iteration D bought a suspected deleveraging shock immediately. G waits for the event opening price to be recovered and invalidates the setup if its low is breached first. This is a state transition, not a retuned indicator period. The hypothesis, thresholds, and rejection criteria were published before implementation in `RECLAIM_MEMO.md`.

## Baseline findings

| Window | Net return | Daily Sharpe | Maximum drawdown | Profit factor | Trades |
| --- | ---: | ---: | ---: | ---: | ---: |
| 2023–2026-11-13 | +3.0803% | 0.2986 | −3.6943% | 1.3016 | 44 |
| 2025–2026-11-13 temporal holdout | −2.2808% | −0.6928 | −3.6943% | 0.5568 | 18 |

The holdout is already exposed historical data, not pristine unseen OOS. Initial capital is 10,000 USDT, with 0.5% risk budgeting and a liquidity cap; these are research portfolio returns, not the running bot's fixed-lot performance. Costs are 10 bps fee plus 5 bps slippage each side. Spot funding is zero. Daily Sharpe includes idle cash days and assumes zero cash yield.

Only 2 of 19 sensitivity cases (baseline included) had positive holdout returns. Doubled friction produced −3.1652%; an extra one-hour metric delay produced −2.3010%, and two hours produced −2.4255%. None supports deployment. The final data-gap guard also has a dedicated regression test.

## Equity curve and trade distribution

The holdout lost 86.82 USDT from trades entered in 2025 and 141.25 USDT from trades entered in 2026. There is no recent-year recovery hidden by aggregation. Eighteen trades is too few to estimate a stable Sharpe or infer conditional alpha.

Seven high-volatility trades lost 262.17 USDT. Eleven low-volatility trades made 34.10 USDT in aggregate, but that includes one 143.57 USDT time exit. Sixteen intrabar stop exits lost 353.89 USDT and one gap-stop exit lost 17.76 USDT. Stops include trailing stops, so a stop exit alone does not imply price breached the original event low. The observed PnL still shows that recovery confirmation frequently fails to produce a sufficiently persistent move.

Removing the best ceil(5%) trades leaves full-sample PnL at −300.70 USDT and holdout PnL at −371.64 USDT. A few favorable extensions explain the positive full-history result. The low absolute drawdown is not a sufficient reason to accept a sparse, negative-expectancy holdout.

`output/structural_reclaim/review.png` plots the holdout equity, drawdown, and each realized trade. Per-trade CSVs and equity series are in that directory. Complete parameter, doubled-cost, and delayed-availability results are recorded in `structural_reclaim_evaluation.json`.

## Self-critique and decision

Reject G. Do not select the profitable low-volatility or prior-day-down subsets after looking at this sample. Such a filter would reuse the same holdout and turn diagnostics into selection bias. A forced seller can explain the initial displacement without leaving exploitable profit after public metrics arrive and price has recovered. OI shrinkage cannot distinguish liquidation from ordinary position closing, and hourly snapshots erase the order-flow sequence.

A materially different follow-up would measure seller absorption directly: signed executed volume, price impact per unit of aggressive selling, and replenishment of bid liquidity at the event. That requires timestamped trades and historical depth or a carefully limited trade-only proxy. Existing 5-minute taker ratios cannot reconstruct hourly signed volume by averaging ratios, and current depth cannot be backfilled into historical events. This is a data requirement and a future hypothesis, not a strategy already validated here. No additional hypothesis was tuned against this holdout in this iteration.

The production strategy, Terra's weight, Telegram heartbeat, and VPS process were not modified.
