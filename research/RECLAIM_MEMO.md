# G: deleveraging followed by structural price reclaim

Specification published in the conversation before implementation, 2026-11-15.

## 1. The Structural Hypothesis
Margin-constrained leveraged longs must sell during adverse price moves. Temporary supply pressure may end before fair value recovers, but OI contraction does not prove liquidation. Unlike the rejected immediate dip-buying hypothesis D, require recovery of the event opening price before entry. The counterparty's constraint explains the original discount; whether any opportunity remains after confirmation is precisely what the test can falsify.

Source: [ECB, Decrypting financial stability risks in crypto-asset markets](https://www.ecb.europa.eu/press/financial-stability-publications/fsr/special/html/ecb.fsrart202205_02~1cc6b111b4.en.html). The reclaim prediction is our inference, not a result established by this source.

## 2. Feature Engineering & Regime Filter
Three entry features: hourly log return / prior-week standard deviation <= -2; OI quantity log change / prior-week standard deviation <= -2; observed closing price >= event opening price. Weekly standardization uses 168 prior hourly observations, excluding the event. Event time is mapped from the conservatively delayed metrics as in prior research. No account-ratio or taker-ratio filter is used. An armed event expires after 24 hours (one daily liquidity cycle), or on any subsequent low touching/breaching the event low. A pending event cannot be replaced with a later, lower stop. Missing data disarms it.

## 3. Execution & Risk Logic
Only available data can arm a candidate. On a later closed hour confirming reclaim, buy spot at next 15-minute open. Original event low is the initial stop; risk budget 0.5% equity, maximum one cash-funded position, size capped at 0.1% of prior 15-minute volume. Trailing stop uses twice prior-day GK volatility, monotonically tightened using closed prices. Maximum holding one week, allowing recovery to develop over several sessions. Fees 10 bps plus slippage 5 bps each side, zero spot funding. Gap stops can exceed the risk budget; it is not a guaranteed loss cap.

## 4. Falsification Criteria
2023–2024 development; 2025–2026-11-13 exposed-history temporal holdout, not pristine OOS. Require holdout daily Sharpe >=1.5, PF >=1.3, drawdown <=15%, >=30 trades, positive PnL removing best ceil(5%) trades. All individual and joint +/-20% parameter perturbations must remain positive, as must doubled fees/slippage and extra 1/2-hour metric availability delays. No parameter selection from holdout. Failure rejects this specification rather than authorizing a profitable subset chosen after observing losses.
