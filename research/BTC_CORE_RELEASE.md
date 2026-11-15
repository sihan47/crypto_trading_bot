# Prepared deployment choice: BTC council core, BNB exit-only

This is a conservative operational choice among existing implemented profiles, not a claim of proven optimal returns. It preserves the user's multi-strategy council and 40% Terra preference, with no new research archetype activated. It does not introduce a new retail-indicator research strategy.

## Exact configuration

`config.btc_core.yaml`:

- BTCUSDT spot, closed 4h bars, 1,000-bar lookback, existing six-expert council and unchanged regime group weights.
- At most one strategy-owned BTC lot, base entry 0.0005 BTC. Terra can veto/reduce; current missing-feed cap prevents increases above 1.0x. This is a quantity cap, NOT a guaranteed percent-risk stop or guaranteed maximum loss.
- BNB council remains present with `entry_enabled: false`: no new BUY, but normal managed SELL eligibility continues. It does not immediately liquidate BNB or abandon existing council-owned lots. A lot from a different strategy scope still requires the existing reconciliation guard.
- Technical 60% / Terra 40%; model `gpt-6.1-terra`, reasoning low, max tokens 2048. News `gpt-5-nano`, 30-minute cache.
- Poll every 60 seconds, evaluate each closed 4h bar once, GPT only when eligible. UTC four-hour Telegram heartbeat remains enabled independently.
- Existing no-add, no-short, balance, exchange lot/notional, duplicate-order and managed-position protections stay in force.

## Evidence and limits

`pruned_council_evaluation.json` reports the BTC technical-only full-allocation historical proxy (2023–2026-11-13): +278.43%, maximum 15m drawdown −25.83%, daily Sharpe 1.224, 90 round trips. Doubled cost: +188.88%, drawdown −33.02%, Sharpe 1.006. These are NOT returns for the fixed-size 0.0005 BTC profile and exclude all GPT decisions. BTC's independently reset 2025 result was approximately −0.20%; returns were not uniformly strong across years.

The BNB pruned council had much deeper historical drawdown (−48.92%) and +60.14% return, motivating no new BNB council exposure. This is an allocation judgment, not a new matched-capital portfolio test. Simpler historical strategies exist with different return/drawdown tradeoffs; this profile is not claimed to beat every one of them.

Research A refinements still have concentrated winners and bootstrap intervals including zero. They are not activated. The latest Terra overlay itself has no historical end-to-end performance validation. Its exact JSON maps approval to a binary support vote, not the old graded confidence. Live anomaly-percentile and authoritative macro-calendar feeds are absent, explicitly marked unknown; no claim of automatic scheduled-release detection is made.

Cost assumptions should not rely on an unverified maker discount. Binance's [regular spot fee table](https://www.binance.com/en/fee/trading) lists 0.100% maker/taker before applicable discounts; actual account fees can differ. No account secrets were inspected for this recommendation.

## Manual activation

From the local repository:

```powershell
powershell -NoProfile -File scripts/deploy_vps.ps1 -Deploy -BtcCore
```

This updates Git on VPS, restarts the bot and selects `BOT_CONFIG=config.btc_core.yaml`. It can submit real orders afterward. It does not rewrite the ignored local/VPS `config.yaml` or `.env`: both use the identical tracked profile. Normal later `-Deploy` preserves the selected profile. `-Recommended` explicitly switches back to the prior BTC+BNB entry-enabled council profile.

No activation/restart was performed when preparing this recommendation. Offline verification covers configuration routing, rejecting non-boolean entry flags, BUY suppression and continued SELL processing.
