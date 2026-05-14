import os
import sys
import time
import pathlib

import pandas as pd
import yaml
from dotenv import load_dotenv
from binance.client import Client
from loguru import logger

from strategy_loader import load_strategy
from trading.order_executor import execute_order, get_portfolio_value, assets_from_symbol
from trading.notifier import send_message
from trading.performance_tracker import record_trade, record_snapshot, summary as perf_summary

# --- Environment ---
load_dotenv()
print("OPENAI_API_KEY loaded:", os.getenv("OPENAI_API_KEY") is not None)

# --- Logging (console + file) ---
pathlib.Path("logs").mkdir(parents=True, exist_ok=True)
logger.remove()
logger.add(
    sys.stdout,
    level="INFO",
    format="{time:YYYY-MM-DD HH:mm:ss} | {level:<8} | [BOT] {message}",
)
logger.add(
    "logs/bot.log",
    rotation="1 day",
    retention="30 days",
    level="INFO",
    format="{time:YYYY-MM-DD HH:mm:ss} | {level:<8} | [BOT] {message}",
)

# --- Helpers ---
def _parse_bool(value, default):
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "on"}
    if value is None:
        return default
    return bool(value)


CONFIG_PATH = pathlib.Path("config.yaml")


def _load_config() -> dict:
    with open(CONFIG_PATH, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def _build_runtime(cfg: dict) -> dict:
    """Parse a fresh config dict into runtime state. Returns the new state to swap in."""
    strat_name = (cfg.get("strategy", {}).get("name") or cfg.get("strategy") or "gpt")
    data_c = cfg.get("data", {}) or {}
    trade_c = cfg.get("trading", {}) or {}
    tf = data_c.get("timeframe", "15m")
    lb = int(data_c.get("lookback", 200))
    global_qty = float(trade_c.get("order_quantity", 0.001))

    symbols_cfg = cfg.get("symbols") or []
    if not symbols_cfg:
        symbols_cfg = [{"symbol": data_c.get("symbol", "BTCUSDT"), "order_quantity": global_qty}]

    runners = []
    for entry in symbols_cfg:
        sym = str(entry.get("symbol", "")).upper()
        qty = float(entry.get("order_quantity", global_qty))
        if not sym:
            continue
        if qty <= 0:
            raise ValueError(f"order_quantity for {sym} must be positive")
        runner = load_strategy(symbol_override=sym)
        runners.append({"symbol": sym, "order_quantity": qty, "strategy_func": runner})

    if not runners:
        raise RuntimeError("No symbols configured. Check config.yaml `symbols` section.")

    assets = sorted({a for s in runners for a in assets_from_symbol(s["symbol"])})
    return {
        "strategy_name": strat_name,
        "timeframe": tf,
        "lookback": lb,
        "symbol_runners": runners,
        "portfolio_assets": assets,
        "trading_cfg": trade_c,
    }


# --- Initial load ---
CONFIG = _load_config()
RUNTIME = _build_runtime(CONFIG)
_config_mtime = CONFIG_PATH.stat().st_mtime

strategy_name = RUNTIME["strategy_name"]
timeframe = RUNTIME["timeframe"]
lookback = RUNTIME["lookback"]
SYMBOL_RUNNERS = RUNTIME["symbol_runners"]
PORTFOLIO_ASSETS = RUNTIME["portfolio_assets"]

# --- Binance client ---
api_key = os.getenv("BINANCE_API_KEY")
secret_key = os.getenv("BINANCE_SECRET_KEY")
testnet_env = os.getenv("BINANCE_TESTNET")
if testnet_env is not None:
    use_testnet = _parse_bool(testnet_env, default=True)
else:
    use_testnet = _parse_bool(RUNTIME["trading_cfg"].get("testnet"), default=True)
client = Client(api_key, secret_key, testnet=use_testnet)


def _maybe_reload_config():
    """Re-read config.yaml if its mtime changed. Swaps in new SYMBOL_RUNNERS atomically."""
    global CONFIG, RUNTIME, SYMBOL_RUNNERS, PORTFOLIO_ASSETS, strategy_name, timeframe, lookback, _config_mtime
    try:
        mtime = CONFIG_PATH.stat().st_mtime
    except Exception as e:
        logger.warning(f"Could not stat config.yaml: {e}")
        return
    if mtime == _config_mtime:
        return
    try:
        new_cfg = _load_config()
        new_rt = _build_runtime(new_cfg)
    except Exception as e:
        logger.error(f"Config reload failed (keeping previous): {e}")
        _config_mtime = mtime  # avoid spamming reload attempts
        return
    old_syms = [s["symbol"] for s in SYMBOL_RUNNERS]
    CONFIG = new_cfg
    RUNTIME = new_rt
    SYMBOL_RUNNERS = new_rt["symbol_runners"]
    PORTFOLIO_ASSETS = new_rt["portfolio_assets"]
    strategy_name = new_rt["strategy_name"]
    timeframe = new_rt["timeframe"]
    lookback = new_rt["lookback"]
    _config_mtime = mtime
    new_syms = [s["symbol"] for s in SYMBOL_RUNNERS]
    logger.info(f"🔄 Config reloaded | symbols: {old_syms} → {new_syms}")
    send_message(f"🔄 Config reloaded | {new_syms}")


def _format_portfolio(snapshot: dict) -> str:
    """Pretty one-line breakdown: asset balances + total USDT equivalent."""
    parts = []
    for asset, info in snapshot["breakdown"].items():
        bal = info["balance"]
        val = info["value_usdt"]
        if asset == "USDT":
            parts.append(f"USDT={bal:.2f}")
        else:
            parts.append(f"{asset}={bal:.6f}(${val:.2f})")
    return f"{' | '.join(parts)} | Total≈${snapshot['total_usdt']:.2f}"


def fetch_live_ohlcv(symbol: str, interval: str, limit: int) -> pd.DataFrame:
    klines = client.get_klines(symbol=symbol, interval=interval, limit=max(2, limit))
    df = pd.DataFrame(
        klines,
        columns=[
            "open_time","open","high","low","close","volume",
            "close_time","qav","num_trades","taker_base","taker_quote","ignore"
        ],
    )
    df = df[["open_time","open","high","low","close","volume"]].copy()
    df["open_time"] = pd.to_datetime(df["open_time"], unit="ms", utc=True)
    df = df.astype({"open": float,"high": float,"low": float,"close": float,"volume": float})
    df = df.set_index("open_time").sort_index()
    return df


def run_one_symbol(sym_cfg: dict, force: bool = False):
    symbol = sym_cfg["symbol"]
    order_quantity = sym_cfg["order_quantity"]
    strategy_func = sym_cfg["strategy_func"]

    df_all = fetch_live_ohlcv(symbol, timeframe, limit=max(lookback + 2, 300))
    closed_df = df_all

    if closed_df.empty:
        logger.warning(f"[{symbol}] No closed bars available.")
        return

    logger.info(f"[{symbol}] Data mode: live | source: Binance | bars={len(closed_df)}")

    if len(closed_df) < 60:
        logger.warning(f"[{symbol}] Not enough bars for indicators (got {len(closed_df)}). Skipping.")
        return

    portfolio_before = get_portfolio_value(PORTFOLIO_ASSETS)
    logger.info(f"[{symbol}] BEFORE | {_format_portfolio(portfolio_before)}")
    send_message(f"[{symbol}] BEFORE | {_format_portfolio(portfolio_before)}")

    try:
        result = strategy_func(closed_df, force=force)
    except TypeError:
        try:
            result = strategy_func(closed_df, initial_run=force)
        except TypeError:
            result = strategy_func(closed_df)

    decision = None
    confidence = None
    reason = None
    regime = None
    if isinstance(result, dict):
        decision = (result.get("last_signal") or result.get("signal") or "").upper()
        confidence = result.get("confidence")
        reason = result.get("reason")
        regime = result.get("regime_label")
    elif isinstance(result, str):
        decision = result.strip().upper()

    extra = []
    if regime:
        extra.append(f"Regime={regime}")
    if confidence is not None:
        extra.append(f"Confidence={confidence}")
    if reason:
        extra.append(f"Reason={reason}")
    extra_part = " | ".join(extra)
    if extra_part:
        extra_part = f" | {extra_part}"
    logger.info(f"[{symbol}] Strategy={strategy_name} | Decision={decision or 'HOLD'}{extra_part} | LastBar={closed_df.index[-1]}")

    if decision in ("BUY", "SELL"):
        order = execute_order(decision, symbol=symbol, quantity=order_quantity)
        if order:
            logger.success(f"[{symbol}] Order executed: {decision}")
            portfolio_after = get_portfolio_value(PORTFOLIO_ASSETS)
            delta = portfolio_after["total_usdt"] - portfolio_before["total_usdt"]
            sign = "+" if delta >= 0 else ""
            logger.info(f"[{symbol}] AFTER  | {_format_portfolio(portfolio_after)} | Δ {sign}${delta:.2f}")
            send_message(f"[{symbol}] AFTER  | {_format_portfolio(portfolio_after)} | Δ {sign}${delta:.2f}")
            record_trade(
                symbol=symbol,
                side=decision,
                quantity=order_quantity,
                portfolio_before=portfolio_before,
                portfolio_after=portfolio_after,
                meta={
                    "confidence": confidence,
                    "regime": regime,
                    "reason": (reason or "")[:200],
                    "order_id": order.get("orderId") if isinstance(order, dict) else None,
                },
            )
        else:
            logger.error(f"[{symbol}] Order failed or was not created: {decision}")
            send_message(f"[{symbol}] ⚠️ Order FAILED: {decision} (not recorded)")


_cycle_count = 0


def run_all_symbols(force: bool = False):
    global _cycle_count
    _maybe_reload_config()
    for sym_cfg in SYMBOL_RUNNERS:
        try:
            run_one_symbol(sym_cfg, force=force)
        except Exception as e:
            logger.exception(f"[{sym_cfg['symbol']}] Error during strategy run: {e}")

    # One snapshot per cycle (after all symbols processed).
    try:
        snap = get_portfolio_value(PORTFOLIO_ASSETS)
        record_snapshot(snap, label="cycle")
        _cycle_count += 1
        # Every 6 cycles (~1 hour at 10-min cadence), refresh the summary file and log it.
        if _cycle_count % 6 == 0:
            s = perf_summary()
            pnl = s.get("pnl_usdt")
            pnl_pct = s.get("pnl_pct")
            trades = s.get("total_trades", 0)
            dd = s.get("max_drawdown_pct")
            if pnl is not None:
                logger.info(
                    f"PERF | trades={trades} | PnL={pnl:+.2f} USDT ({pnl_pct:+.2f}%) | maxDD={dd:.2f}%"
                )
    except Exception as e:
        logger.warning(f"Performance snapshot failed: {e}")


if __name__ == "__main__":
    symbols_summary = ", ".join(f"{s['symbol']}(qty={s['order_quantity']})" for s in SYMBOL_RUNNERS)
    logger.info(f"Bot started | strategy={strategy_name} | symbols=[{symbols_summary}] | timeframe={timeframe} | lookback={lookback}")
    try:
        initial_portfolio = get_portfolio_value(PORTFOLIO_ASSETS)
        logger.info(f"Initial portfolio | {_format_portfolio(initial_portfolio)}")
        send_message(f"🚀 Bot started | {_format_portfolio(initial_portfolio)}")
    except Exception as e:
        logger.warning(f"Could not fetch initial portfolio: {e}")
    logger.info("First forced run")
    run_all_symbols(force=True)

    while True:
        run_all_symbols(force=False)
        time.sleep(10 * 60)
