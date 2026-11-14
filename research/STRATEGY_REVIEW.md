# Strategy review ? 2026-11-15

## Decision

Candidate: 4h regime ensemble, base weights trend 40%, breakout 30%, range reversion 30%. Deploy in **shadow mode** first; no real orders. The GPT arbiter is GPT-6.1 Terra with 20% bounded review influence (switched back after account usage confirmed incentive-tier requests). Do not interpret scores as probabilities.

## Research basis

- [Moskowitz, Ooi, Pedersen / AQR: time-series momentum](https://www.aqr.com/Insights/Datasets/Time-Series-Momentum-Original-Paper-Data): motivation for a trend family. Their monthly, diversified futures evidence is NOT proof of an intraday crypto edge.
- [John Bollinger: original band rules](https://www.bollingerbands.com/bollinger-band-rules): touching a band alone is not a buy/sell signal. The candidate requires re-entry into the lower band in a range, with RSI confirmation.
- [Turtle rule documentation](https://www.turtletrader.com/rules/): motivation for separate breakout entry/exit windows. This implementation is an adaptation, not a reproduction of the original daily futures system.
- [Bailey et al.: Probability of Backtest Overfitting](https://www.davidhbailey.com/dhbpapers/backtest-prob.pdf): limit candidate count, retain losing results, separate selection and later evaluation.

## Protocol and limitations

- Public Binance spot BTCUSDT/BNBUSDT data from 2022-09 through 2026-11-13. Complete 15m bars are built from local 1m data (left-labeled, left-closed), then appended from the public 15m endpoint. 1h/4h bars require all underlying 15m bars. A known 2023-03-24 outage remains a gap, never interpolated.
- Fixed indicator parameters: EMA 48/192, Donchian entry 96 / exit 48, Bollinger 20/2, RSI 14, ADX 14. Units are bars of the selected timeframe.
- Four preset weight triples per timeframe: balanced (0.4/0.3/0.3), trend (0.6/0.3/0.1), breakout (0.3/0.6/0.1), range (0.2/0.2/0.6). No exhaustive search.
- Ranking uses average return plus half the negative max drawdown across BTC/BNB, 2023, 2024 and 2025. The 2026 period is excluded from weight ranking. The 4h winner also has the highest development/validation ranking across these three horizons.
- **Research iteration disclosure:** 1h/4h were added after viewing the failed 15m results. The later 2026 evaluation is chronological out-of-sample relative to weight selection, but NOT a pristine blind final test of the entire research process. New forward shadow observations are needed.
- Signals at a closed bar execute at the NEXT bar open. Both sides pay 0.10% fees + 0.05% slippage. No same-close fills. Open positions are liquidated with costs at each fold end. Each fold starts in cash.
- These results simulate a single fully allocated long position. Live sizing stays at 0.0005 BTC / 0.05 BNB per alternating signal, so **these percentages are not a forecast of the real account**. Existing holdings are not automatically fully liquidated.
- Historical GPT/news decisions are not recreated. GPT review, account balances, rejected orders and API outages are not included in performance figures.
- Existing 15m baselines use the existing best_params.json, which was fitted through 2025/May 2026. Development results for those baselines are biased. 1h/4h baselines use default parameters, not the 15m fitted parameters.
- Small trade counts and researcher choices limit confidence. No profitability or significance claim.

## Later-period comparison: 2026-05-02 to 2026-11-13

| Timeframe / selected weights | BTC return | BTC drawdown | BTC round trips | BNB return | BNB drawdown | BNB round trips |
|---|---:|---:|---:|---:|---:|---:|
| 15m / range_focus | -7.50% | -7.62% | 35 | -12.42% | -13.07% | 41 |
| 1h / trend_focus | 2.78% | -17.57% | 18 | -18.50% | -25.93% | 21 |
| 4h / balanced | 3.55% | -9.09% | 9 | 7.52% | -15.85% | 9 |

## Selected 4h candidate details

| Period | BTC return / drawdown / trades | BNB return / drawdown / trades |
|---|---|---|
| development_2023 | 30.12% / -18.15% / 26 | -0.34% / -28.44% / 17 |
| development_2024 | 53.18% / -29.26% / 19 | 43.50% / -23.90% / 17 |
| validation_2025 | 3.70% / -14.34% / 16 | 10.42% / -23.56% / 13 |
| holdout_2026 | 3.55% / -9.09% / 9 | 7.52% / -15.85% / 9 |

Double-cost stress (0.20% fee + 0.10% slippage on each side):
- BTCUSDT: return 0.79%, drawdown -10.13%. Ordinary-cost buy-and-hold benchmark: -13.13% return, -29.25% drawdown.
- BNBUSDT: return 4.66%, drawdown -16.57%. Ordinary-cost buy-and-hold benchmark: 1.99% return, -26.67% drawdown.

## Runtime changes

- New strategy families replace the eight overlapping votes when engine=regime_v2. Legacy code remains available for comparison; it is not used by the new configuration.
- Regime-specific weights: in ranges, trend/breakout multiply by 0.5 and reversion by 2; in trends, reversion multiplies by 0.25. Scores divide by active family weight; at least 25% total weight must participate. Entry threshold 60, exit 55, directional margin 10.
- ATR blocks new buys only. Existing positions can generate SELL during high volatility. GPT cannot create a trade from local HOLD or bypass allowed actions; a 20% review can weaken/confirm a local trade. API/schema errors hold.
- Only closed bars; at most one evaluation per symbol/timeframe/bar across restarts. No second immediate startup evaluation. Missing account data fails closed.
- Same-side submitted trades are suppressed until an opposite confirmed trade. Intent is persisted before submitting an order; unknown outcomes block future orders pending reconciliation. This journal assumes a single bot process.
- Shadow mode logs SHADOW_ORDER and keeps shadow state separate from live orders. It does not simulate fills, fees or changing balances and is not a paper PnL engine.
- A successful exchange order is no longer reported failed merely because a subsequent balance query failed.
- Failure to evaluate a claimed bar is not automatically retried on that bar; the next closed bar is evaluated. This prioritizes avoiding repeated submissions.
- The 4h candidate is a slow strategy: nine historical round trips over roughly six months is not a promise of daily trades.

## Verification

- Offline tests cover causality, next-open execution, costs, warm-up, active-weight normalization, ATR exits, GPT restrictions, persistence, shadow execution and config wiring.
- Six rolling 1000-bar window checks per symbol matched the full-history final entry/exit signals at sampled dates.
- Public latest-bar diagnostic plus actual Sol review produced SELL for both symbols in a synthetic in-position scenario. This was a read-only test; no real order was submitted.

## Reproduce

```powershell
python research/prepare_ensemble_data.py --end 2026-11-14
python research/evaluate_ensemble.py --timeframe 15m
python research/evaluate_ensemble.py --timeframe 1h
python research/evaluate_ensemble.py --timeframe 4h
python -m unittest discover -s tests -p test_ensemble.py
python -m unittest discover -s tests -p test_decision_controls.py
python -X utf8 -m unittest discover -s tests -p test_runner_shadow.py
python scripts/check_ensemble.py --review
```

Raw metrics, including all losing variants: ensemble_evaluation.json, ensemble_evaluation_1h.json, ensemble_evaluation_4h.json.
For an unresolved live intent in logs/decision_state.json, reconcile against exchange order history before modifying the pending record; never simply delete the journal to force another order.
