# Frozen IS tournament specification

The four-part research memo was published in the conversation before strategy implementation. This file records that specification and implementation details; no candidate was selected using the final 30%.

## 1. Structural hypotheses

A: in a positive 90-day price regime, low standardized funding and positive standardized top-position/global-account divergence may identify crowded shorts or reluctant leveraged longs while larger position holders lean long. A squeeze could generate convex continuation. Funding below its historical mean is NOT necessarily a negative absolute funding payment. Position/account ratios do not identify smart money or prove who is forced to cover.

B: urgent short-term sellers may pay for immediacy in a low-impact/high-damping state; a passive buyer might earn reversion toward recent transaction prices. This can fail through informed selling, adverse selection, fill uncertainty and fees. Earlier volatility associations were partly explained by confounders; they are hypotheses, not established directional alpha.

C: margin-constrained long holders may sell rapidly during a joint price/OI flush. A subsequent spot buyer might capture recovery, but OI declines do not prove forced liquidation. The impulse scan found too few events for inference; C remains an explicitly low-evidence benchmark, not a validated edge. The exhaustion branch had zero events and is not implemented as a short strategy.

## 2. Features and state

A uses three conditions: 90-day bull state, `oi_weighted_funding < 0`, and `position_account_divergence > 0`. Features retain the published EDA definitions. This conjunction differs from the original funding decile statistic and must be tested on its own.

B uses either tail (past 90-day 10/90% thresholds) of either published damping feature, plus a spot price below the prior-completed four-hour close-volume-weighted center minus one 24-hour ATR. The center is an hourly close-volume proxy, NOT tick-accurate VWAP. It rests a bid at the lesser of that level and current close less 5 bps. It does not use a generic price indicator as its standalone primary trigger.

C uses the union of fixed 1h and 3h liquidation proxy events: event-aligned price loss over 2.5 prior 24h ATR, OI contraction over 3%, and taker flow below its prior 30-day first percentile. Metric timestamps and event-hour prices are aligned using the existing conservative publication delay. The current observed spot low must still exceed the event low. No profitable subset of impulse horizons was chosen.

## 3. Execution and risk

All three buy unleveraged BTC spot; no shorts, borrowing, perpetual inventory or funding payments. Derivatives data is information only. Initial capital 10,000 USDT, risk budget 0.5% per trade, one position, cash cap, and previous-bar volume cap. Stop gaps may lose more than the risk budget.

A: next 15-minute open plus 5 bps slippage, initial stop at two daily GK sigmas below decision close, two-sigma trailing stop updated using closed prices, maximum 168 hours.

B: post-only approximation rejects quotes at/above the submission open. Pending order expires after four 15-minute bars. Fill requires trade-through by more than 5 bps, not touch; no favorable gap improvement. Size is capped at 0.01% of previous 15-minute volume (other archetypes 0.1%). Initial stop is two ATR below the quote, target is the fixed pre-submission center, maximum hold four hours. Stop wins ambiguous bars; no target exit on the fill bar. Exit is conservatively charged as taker. Entry pays the baseline 10 bp fee plus a 5 bp execution reserve rather than pretending a limit can fill above its limit. The reserve appears in the ledger's `fees` cost column and is not an asserted exchange fee.

C: next-open spot entry, event low stop, maximum 12 hours, no fixed profit target or optimized reversal filter.

Base market fee 10 bps plus 5 bps slippage per side. B separately evaluates a HYPOTHETICAL 6 bp maker fee; account eligibility is not assumed. A stricter 10 bp penetration scenario is also reported. Without historical quotes, queue positions and trades at price, neither OHLC penetration rule establishes actual maker fills. B performance is conditional simulation, not a queue-validated backtest. Binance explicitly notes that [a limit order need not be maker](https://www.binance.com/en/support/faq/detail/360007720071); actual fees depend on the [account fee tier](https://www.binance.com/en/fee/trading).

## 4. Falsification and walk-forward design

Only the earlier 70% discovery partition is used. One year initializes causal history, followed by chronological 90-day evaluation windows and a final partial window. Rules and parameters are frozen: no grid-search winner is fit. Rolling normalizations update using only past data. Cash/positions reset per window; fold returns are compounded and trade cash PnLs rescaled to the resulting capital. Forced fold-end exits pay costs. This is fixed-spec internal walk-forward evaluation, not fresh OOS, because EDA already examined these dates.

Report Sharpe and Sortino on calendar daily equity including cash days, zero cash yield, annualization sqrt(365); Sortino denominator is RMS negative daily returns about zero. PF, win rate, average net trade PnL and counts use realized trades. Sparse C estimates are descriptive only.

Sensitivity perturbs each applicable risk/stop/trailing/deviation/impulse/tail/holding/liquidity parameter +/-20%, plus joint corners. EDA normalization and macro-regime calendar horizons are retained, so this is not exhaustive sensitivity over every feature construction choice. Double all fees/execution friction. Require at minimum positive stressed returns and enough trades before further validation; original Sharpe >=1.5, PF>=1.3, >=30 trades, DD<=15%, and positive results removing best 5% remain reference qualification criteria. Neither a positive IS return nor a positive subset licenses deployment.

This run does not consume final reserved-period prices, estimate final OOS performance, or alter the running bot.
