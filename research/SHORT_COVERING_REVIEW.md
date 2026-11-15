# Structural research E/F — 2026-11-15

## Decision

Neither candidate qualifies for production. No strategy, GPT weight, live configuration, or running process was changed by this research.

## E: short-covering continuation

The pre-test specification is in `SHORT_COVERING_MEMO.md`; implementation and complete results are in `structural_short_covering.py` and `structural_short_covering_evaluation.json`.

| Period / stress | Net return | Daily Sharpe | Trades |
| --- | ---: | ---: | ---: |
| Development, 2023–2024 | +1.3616% | 0.7312 | 4 |
| Temporal holdout, 2025–2026-11-13 | −0.8430% | −1.3385 | 2 |
| Holdout, doubled friction | −0.9480% | −1.4059 | 2 |
| Holdout, additional one-hour data delay | −0.8572% | −1.1425 | 2 |
| Holdout, additional two-hour data delay | −0.9975% | −1.0548 | 2 |

Only 1 of 21 sensitivity cases (including the baseline) had positive holdout returns. Holdout profit factor was zero. Maximum drawdown was 0.8430%, reflecting very little exposure rather than evidence of attractive risk-adjusted returns. Both trades stopped out: July 6–7 lost 34.47 USDT and October 11–12 lost 49.83 USDT on the 10,000 USDT research account. The equity curve therefore has only two losing excursions, separated by long periods in cash.

Both losses occurred in the low-volatility diagnostic regime following an upward prior day. This is insufficient evidence to conclude that removing that regime creates an edge: doing so would remove the entire holdout sample. Price rising while OI falls and taker buying dominates can mark the exhaustion of buying pressure, not its continuation. OI and taker ratios do not identify liquidations or the counterparties directly. Reject this continuation hypothesis; do not rescue it with a post-hoc regime filter.

Spot execution uses next-bar entry, structural stops, 0.5% equity risk, and 0.1% fee plus 0.05% slippage per side. Spot funding is zero. Metrics availability includes the existing conservative publication delay. This historical period has already informed earlier research and is not a pristine unseen out-of-sample test. Two trades cannot establish a stable Sharpe estimate.

## F: positive spot/perpetual basis convergence

The structural pivot changes the source of returns from directional continuation to compensation for providing two-legged liquidity. See the pre-screen specification in `BASIS_FEASIBILITY_MEMO.md` and machine-readable results in `basis_dislocation_screen.json`.

Round-trip fees and slippage across both legs total 52 basis points; a 1.25 cost buffer requires 65 bps of positive basis. The highest observed positive hourly close basis was 28.88 bps in development and 6.28 bps in holdout. There were **zero candidate hours** at baseline, either ±20% hurdle, or doubled friction. It fails the predeclared minimum of 30 day-separated holdout events.

This is a cost feasibility screen, not a PnL backtest: no returns, Sharpe, funding profits, or simulated fills are claimed. It rejects this positive-basis, taker-cost, hourly-close implementation only. It does not rule out intrahour dislocations, rigorously modeled maker execution, or negative-basis trades with verified spot borrow costs. Funding carry was evaluated separately in iteration C. Hourly close differences are not executable simultaneous bid/ask quotes.

## Operational verification

Read-only VPS inspection found commit `8ea1671` running, with the persistent heartbeat state acknowledging the 2026-11-15 16:00 UTC window after a successful Telegram response at 16:03:17 UTC (17:03:17 Berlin). The next scheduled window is 20:00 UTC / 21:00 Berlin, subject to completion of the bot loop. This verifies delivery acknowledgement, not that the recipient read the message.

## Source and reproducibility

OI definitions: [CME Open Interest](https://www.cmegroup.com/education/lessons/open-interest). Source archives: [Binance public data](https://github.com/binance/binance-public-data). The causal trading hypotheses are research inferences, not conclusions asserted by these sources.

Run `python research/structural_short_covering.py` and `python research/screen_basis_dislocation.py` with the locally cached, audited datasets. Raw data and per-trade/equity CSV outputs remain local under the repository's existing ignore rules.
