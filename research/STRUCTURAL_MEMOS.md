# BTC structural research: frozen before strategy implementation

2026-11-15. Existing BTC spot OHLCV history has already been inspected in other research. Temporal holdout is NOT pristine unseen data. No production/config changes or live activation.

## Iteration A

### 1. The Structural Hypothesis (Why does this edge exist?)
Compression can allow volatility-targeted leverage to build; an upward shock may force short covering and risk rebalancing. Counterparty pain would be margin/risk constraints, not an oscillator. This is an unverified mechanism: OHLCV cannot identify forced counterparties. Test upward continuation following compression and participation expansion, not generic price-band crossing.

### 2. Feature Engineering & Regime Filter
Closed hourly bars. GK variance v = .5 log(H/L)^2 - (2 log(2)-1) log(C/O)^2, clipped at zero for numerical safety. Three features: previous 24h/168h GK volatility ratio q <= .8; current hourly log return divided by previous 24h hourly GK sigma z >= 2; close*volume divided by previous 168h median close*volume u >= 2. Daily/weekly windows reflect risk-management cycles. The .8/2 thresholds are explicit hypotheses, not optimized constants. All reference windows exclude the current bar. Require complete windows.

### 3. Execution & Risk Logic
Spot long only, next available consecutive 15m opening price after hourly confirmation. Fee .001 and slippage .0005 EACH side; repeat with both doubled. No perpetual position, funding identically zero. Planned loss budget .005 equity; no leverage, and order notional <= .001 of previous completed 15m close*volume. Initial stop = previous 24h low; reject invalid stop. Stop only ratchets upward to highest CLOSED 15m close * exp(-2*known daily GK sigma). No fixed profit target, seven-day maximum holding time. Stop gaps fill at worse open; stop is not guaranteed loss cap. Entry-bar stops included. No intrabar high ratchet (avoids OHLC path hindsight). Missing bars: no new entry across gaps; mark uncertainty and exit any held lot at next observed open. Friction is a conservative scenario, not measured order-book impact.

### 4. Falsification Criteria
Development 2023-2024; exposed-history temporal holdout 2025-2026/11/13. Reject recommendation if holdout daily Sharpe <1.5, net trade PF <1.3, marked max drawdown >15%, fewer than 30 round trips, or removing top ceil(5%) winners makes net trade PnL nonpositive. Each parameter (windows, thresholds, stop multiplier, duration, risk, participation cap) independently +/-20%, plus all-parameter +/-20% corners: every holdout return must stay positive. Double-friction holdout must remain positive. No tune-and-retest rescue. Diagnose loss regimes; discard or publish a new structural memo before mutation.

## Data and source limits

Local file `data/research_15m/BTCUSDT.csv`: 147,386 observations, 2022-08-31 16:00 to 2026-11-13 23:45 UTC; one 105-minute timestamp gap (six missing 15m bars). Has OHLCV only: NO historical taker imbalance, OI, liquidation tape, or funding. Dollar volume is close*base volume, not exact quote volume; cannot reconstruct exact dollar bars or claim actual liquidation detection.

Sources consulted:

* Binance public-data definitions: https://github.com/binance/binance-public-data
* Kroner/Mohammed/Vega, 2026 draft, macro announcements and crypto price discovery: https://users.nber.org/~confer/conf_papers/2387/f238700/f238700.pdf . Supports a news/repricing failure mode, not profitability of these strategies.
* Garcia Seuma, 2026 preprint: https://arxiv.org/abs/2608.03616 . Studies actual cascade episodes; warns against simplistic scalar pre-cascade prediction. Limited event set, not evidence that this OHLCV proxy identifies liquidation.

## Iteration A falsification and decision

Development Sharpe 1.70, holdout -0.11, holdout return -0.94%, doubled-cost -3.79%. Only 1/21 sensitivity cases profitable. Low-volatility entries account for -169.73 USDT net (34 trades), versus +75.86 high-volatility (17). Prior-day-up entries lose -171.02. These small descriptive groups are not causal evidence or permission to select a winning filter. Reject A; do NOT adopt the one profitable compression perturbation.

## Iteration B (memo published before implementation)

### 1. The Structural Hypothesis (Why does this edge exist?)
After urgent sell-side liquidation or constrained selling, liquidity providers may earn a discount once price has already reclaimed half the event range. Forced sellers pay immediacy costs. However macro/news repricing can make the fall permanent. OHLCV is only a sell-pressure shock proxy, not proof of actual liquidations; reject if subsequent continuation overwhelms rebounds.

### 2. Feature Engineering & Regime Filter
Three features: hourly return/prior-day GK sigma <= -2; close*volume/prior-week median >=2; recovery location (close-low)/(high-low)>=.5. The half-range criterion is structural recovery, not an optimized oscillator level. Windows remain a day/week, all references lagged, complete closed hourly bars only. Compression is not used.

### 3. Execution & Risk Logic
Spot long, next 15m open; initial invalidation is event-hour low, not prior-day low. Same .5% planned risk, unleveraged notional, .1% prior 15m volume participation cap. No fixed target; same two-daily-sigma trailing risk rule, maximum 24h inventory holding time. Same fees/slippage, gap treatment, zero funding. No claim of guaranteed loss cap.

### 4. Falsification Criteria
Same development/holdout and all A acceptance thresholds. Independently perturb each active B parameter +/-20%, plus common corners. No retrospective session or volatility exclusion to rescue negative results. If failed, discard OHLCV shock-recovery as deployable alpha and require direct flow/OI evidence before a liquidation-specific claim.

## Iteration B falsification and decision

Holdout -3.79%, Sharpe -1.80, PF .081, 11 trades. Eight event-low stop exits lose 392.63 USDT, three time exits earn 13.75. Prior-day-down entries lose 346.09 across eight observations. No sensitivity case profitable (one has no trades). Too few observations for a reliable regime claim; reject rather than remove losing sessions. Pivot to direct funding carry.

## Iteration C (memo published before strategy implementation)

### 1. The Structural Hypothesis (Why does this edge exist?)
Leveraged directional longs pay positive funding for financed perpetual exposure. Buy spot/short equal BTC perpetual to receive that payment, subject to basis/margin/execution risk. Longs are paying for a service, not necessarily losing overall. This is a financing/risk premium hypothesis, not risk-free arbitrage and not an OHLCV liquidation proxy.

### 2. Feature Engineering & Regime Filter
Only two core features: trailing seven-day realized funding per day and perpetual/spot basis F/S-1. Enter only positive funding and nonnegative basis when 30-day forecast funding plus basis exceeds 1.25 times complete round-trip costs. Use realized, not future announced rates, delayed one hour for decision eligibility. A month is a capital commitment horizon; 25% hurdle buffer allows forecast error. Baseline notional .40 of total equity, leaving matched one-times futures margin and cash reserve.

### 3. Execution & Risk Logic
Both legs next hour open after previous close evidence. Spot .001 fee, futures .0006 fee, both .0005 slippage each side. Exit when realized funding expectation turns negative, marked pair loss reaches .01 of entry equity, 30-day holding limit, or isolated-margin/effective-notional <.25. Validate each hour's mark-price HIGH against a conservative 1% maintenance requirement; any possible breach invalidates acceptance rather than claiming guaranteed hedging protection. Actual settlement rates credited/debited only during ownership; do not collect positive funding on entry hour, charge negative entry-hour funding conservatively. Funding mark price from settlement response when present; if absent use corresponding hourly mark OPEN (timestamp jitter must be <60 seconds), explicitly disclose approximation. Missing market data forces exit at next observed open, forbids entry across gap. Sensitivity covers each active parameter +/-20% plus corners, not fee reduction. Stress doubles all friction; separate one-hour perp-leg delay scenario and 4%-annual cash opportunity-cost scenario (assumed hurdle, not current cash yield quote).

### 4. Falsification Criteria
Same Sharpe>=1.5, PF>=1.3, DD<=15%, >=30 holdout round trips, positive after removing best 5%, all parameter returns positive, double-cost positive. Any margin-breach flag fails. Incomplete funding history or unmodelled legging risk prevents qualification. No production/perpetual account changes.
