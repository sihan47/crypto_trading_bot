# Live risk-overlay context

Every actual GPT judge request now assembles public Binance positioning telemetry,
scheduled macro events and the existing NewsAPI news provenance. The four-hour
decision cadence and heartbeat are unchanged. No new strategy is activated.

## Inputs and limits

- Binance closed hourly futures klines include taker-buy volume. Actual settled
  funding, OI, top position/account ratios and global account ratios are included.
  Historical daily metrics archives are SHA256-checked. First use bootstraps 130
  days; subsequent runs reuse disk archives. A cold VPS can take several minutes.
- Feature percentiles use up to 2,160 preceding hourly observations, excluding the
  current observation, with at least 720 valid observations. Missing history is
  reported, not replaced with invented values. Metrics use a conservative 65-minute
  publication buffer; funding uses one hour. These are assumptions, not audited
  publication logs. Latest positioning older than three hours is rejected.
- FOMC dates: Federal Reserve calendar, standard 14:00 New York statement time
  with DST conversion. PCE: BEA ICS. CPI/NFP: BLS ICS when accessible; FairEconomy
  weekly calendar is an explicitly labelled third-party backup. Conflicting
  event times are retained. Coverage is source-reported, not a completeness guarantee.
- News remains the existing NewsAPI + configured summary model. The judge also
  receives source URLs, publication times, retrieval time and age. A 72-hour query
  is not a guarantee of instant breaking-news delivery. No new news API key is needed.
- Fetch errors, warmup and staleness are explicit. A missing source does not mean
  a quiet market. Data caches expire; stale values are not silently promoted to live.

Known FOMC/CPI/NFP/PCE events within four hours deterministically veto BUY even if
the model approves; SELL is not blocked by this scheduled-event guard. Partial
calendar coverage can still veto a known event. Telegram judge reports show the
effective action/size after this guard. The configured 40% Terra weight is unchanged;
the live-context cap remains 1.0x, so connecting feeds does not increase position size.

Microstructure priors remain exploratory BTC associations, not validated trading
alpha or evidence that top traders are smart money. Global accounts are not a pure
retail sample. BNB cannot inherit BTC-specific statistical evidence.

## Verification and deployment

`python scripts/verify_release.py`: explicit offline test suite, including timezone,
calendar freshness, hard-veto behavior, percentile exclusion and failure handling.
`python scripts/check_live_context.py BTCUSDT`: public-data-only smoke test, writes
`output/live_context_probe.json`; no GPT calls, account access, orders or Telegram.

2026-11-15 22:00 UTC smoke: BTC telemetry available with no archive failures;
calendar covered all four event types using the labelled fallback. BLS and the
next-week backup failed HTTP requests. News was deliberately not called in this
public-only probe, so its status was unavailable in that probe.

This change does not restart the VPS. After pushing, the existing manual deployment
command `powershell -NoProfile -File scripts/deploy_vps.ps1 -Deploy -BtcCore`
selects BTC core / BNB exit-only and loads the new integration on restart.
