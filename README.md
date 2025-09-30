# Crypto Trading Bot

Python-based trading bot that combines classic technical indicators with GPT-driven decision making and live BTC news summaries. It targets Binance (testnet by default) and includes utilities for backtesting, parameter tuning, and notifications.

## Key Features
- Live Binance market data ingestion (REST/WebSocket) with testnet safety.
- Modular strategy loader supporting SMA, RSI, MACD, Bollinger, and a GPT ensemble.
- LLM prompt builder that enriches signals with Gemini-powered BTC news summaries.
- Centralized order execution, logging, and Telegram notifications.
- Research tooling for backtesting, signal analysis, and parameter tuning.

## Project Layout
- `run_bot.py` - main entry point for the live bot loop.
- `strategies/` - individual indicator strategies, GPT meta strategy, and news summarizer.
- `trading/` - order execution, performance tracking, notifier, and WebSocket manager.
- `backtesting/` & `research/` - utilities for historical analysis and reporting.
- `config.yaml` - runtime configuration for strategy, data feed, and trade sizing.

## Prerequisites
- Python 3.11 (repo ships conda instructions, but any equivalent environment works).
- Binance account with testnet keys.
- Optional integrations:
  - OpenAI API key for GPT strategy.
  - Google Gemini API key and NewsAPI key for news summaries.
  - Telegram Bot token and chat ID for alerts.

## Setup
```bash
# create environment (optional but recommended)
conda create -n crypto_trading python=3.11 -y
conda activate crypto_trading

# install dependencies
pip install -r requirements.txt
```

Populate `.env` (copy `.env.example` if you maintain one) with:
```
BINANCE_API_KEY=...
BINANCE_SECRET_KEY=...
OPENAI_API_KEY=...
NEWS_API_KEY=...
GEMINI_API_KEY=...
TELEGRAM_BOT_TOKEN=...
TELEGRAM_CHAT_ID=...
```
Keys that are not available can be left unset; related features will degrade gracefully.

## Configuration (`config.yaml`)
```yaml
strategy:
  name: gpt            # gpt | sma | rsi | macd | bollinger
  params:
    context_hours: 24  # hours of OHLCV for GPT prompt
    provider: openai   # openai | mock
    show_prompt: true
    mode: live

data:
  symbol: BTCUSDT
  timeframe: 15m
  lookback: 200

trading:
  order_quantity: 0.001  # trade size in BTC
```
- `strategy.name` selects the runner loaded by `strategy_loader.py`.
- `data` drives live OHLCV fetching; adjust symbol/timeframe to match Binance symbols.
- `trading.order_quantity` controls market order size and must be positive.

## Running the Bot
```bash
python run_bot.py
```
The bot will:
1. Pull recent OHLCV bars from Binance.
2. Collect strategy signals and optional GPT + news context.
3. Execute market orders when the decision is BUY or SELL.
4. Log activity to `logs/bot.log` and notify Telegram (if configured).

Default sleep interval is 10 minutes between evaluations; adjust logic in `run_bot.py` if you need faster cadence.

## Research & Backtesting
- `backtest.py` and modules under `backtesting/` let you simulate indicator strategies.
- `research/parameter_tuner.py` and `research/signal_runner.py` help evaluate parameter grids and generate reports.
- Generated results are stored under `research/signals/` and associated plots.

## Testing & Validation
There is no formal test suite yet. Recommended quick checks:
```bash
# syntax check key modules
python -m compileall run_bot.py trading/order_executor.py strategies/btc_news_summary.py

# dry-run news summarizer
python strategies/btc_news_summary.py
```
Consider adding pytest-based coverage for order execution and strategy loading before deploying to production exchanges.

## Logging & Monitoring
- Log files rotate daily (`logs/bot.log`, `logs/trading.log`, `logs/performance.log`).
- `trading/performance_tracker.py` can be scheduled to compute daily PnL and win-rate stats.
- `trading/ws_manager.py` provides a reusable WebSocket manager for tighter real-time needs.

## Practical Tips
- Keep API keys off source control; rely on environment variables.
- Start on Binance testnet until your strategy and risk controls are validated.
- Rate limit and cache external news/API calls when running long sessions.
- Review `requirements.txt` and lock additional dependencies as features expand.

Happy trading!
