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
from trading.decision_state import DecisionState, closed_bars
from trading.managed_position import managed_position, market_quantity, confirm_managed_order, amount
from trading.order_executor import execute_order, get_portfolio_value, assets_from_symbol
from trading.notifier import send_message
from trading.heartbeat import Heartbeat
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


CONFIG_PATH = pathlib.Path(os.getenv("BOT_CONFIG", "config.yaml"))
DECISION_STATE = DecisionState(pathlib.Path("logs/decision_state.json"))
HEARTBEAT = Heartbeat(pathlib.Path("logs/heartbeat_state.json"))
SYMBOL_STATUS = {}


def _load_config() -> dict:
    with open(CONFIG_PATH, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def _build_runtime(cfg: dict) -> dict:
    """Parse a fresh config dict into runtime state. Returns the new state to swap in."""
    strat_name = (cfg.get("strategy", {}).get("name") or cfg.get("strategy") or "gpt")
    data_c = cfg.get("data", {}) or {}
    trade_c = cfg.get("trading", {}) or {}
    if trade_c.get("execution_mode", "shadow") not in {"live", "shadow"}:
        raise ValueError("execution_mode must be live or shadow")
    poll_seconds = int(trade_c.get("poll_seconds", 600))
    if not 10 <= poll_seconds <= 3600:
        raise ValueError("poll_seconds must be between 10 and 3600")
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
        symbol_tf = str(entry.get("timeframe", tf))
        engine_override = entry.get("engine")
        runner = load_strategy(config_path=str(CONFIG_PATH), symbol_override=sym,
                               timeframe_override=symbol_tf, engine_override=engine_override)
        runners.append({"symbol": sym, "order_quantity": qty, "strategy_func": runner,
                        "timeframe": symbol_tf, "engine": engine_override})

    if not runners:
        raise RuntimeError("No symbols configured. Check config.yaml `symbols` section.")

    assets = sorted({a for s in runners for a in assets_from_symbol(s["symbol"])})
    return {
        "poll_seconds": poll_seconds,
        "strategy_name": strat_name,
        "timeframe": tf,
        "lookback": lb,
        "symbol_runners": runners,
        "portfolio_assets": assets,
        "trading_cfg": trade_c,
        "engine": (cfg.get("strategy", {}).get("params", {}).get("engine", "legacy") if isinstance(cfg.get("strategy"), dict) else "legacy"),
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
    klines = client.get_klines(symbol=symbol, interval=interval, limit=min(1000, max(2, limit)))
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
    status = SYMBOL_STATUS.setdefault(symbol, {})
    previous_health = status.get('health')
    status['health'] = '檢查中'
    order_quantity = sym_cfg["order_quantity"]
    strategy_func = sym_cfg["strategy_func"]
    symbol_timeframe = sym_cfg.get("timeframe", timeframe)

    df_all = fetch_live_ohlcv(symbol, symbol_timeframe, limit=max(lookback + 2, 300))
    closed_df = closed_bars(df_all, symbol_timeframe)

    if closed_df.empty:
        status.update(health='資料異常',reason='沒有已收盤 K 線',decision='未評估',gpt='未呼叫',order='無新訂單')
        logger.warning(f"[{symbol}] No closed bars available.")
        return

    logger.info(f"[{symbol}] Data mode: live | source: Binance | timeframe={symbol_timeframe} | bars={len(closed_df)}")

    if len(closed_df) < 60:
        status.update(health='暖機不足',reason=f'只有 {len(closed_df)} 根 K 線',decision='未評估',gpt='未呼叫',order='無新訂單')
        logger.warning(f"[{symbol}] Not enough bars for indicators (got {len(closed_df)}). Skipping.")
        return

    if not DECISION_STATE.claim_bar(symbol, symbol_timeframe, closed_df.index[-1]):
        status.update(health='執行錯誤' if previous_health=='執行錯誤' else '等待下一根收盤',bar=str(closed_df.index[-1]))
        status.setdefault('reason','這根 K 線已在之前處理；本程序沒有該次決策摘要。')
        logger.info(f"[{symbol}] Closed bar already evaluated: {closed_df.index[-1]}")
        return

    status.update(health='評估中',bar=str(closed_df.index[-1]),decision='未完成',gpt='尚未完成',reason='新 K 線評估中',order='尚未送單')
    portfolio_before = get_portfolio_value(PORTFOLIO_ASSETS)
    logger.info(f"[{symbol}] BEFORE | {_format_portfolio(portfolio_before)}")
    send_message(f"[{symbol}] BEFORE | {_format_portfolio(portfolio_before)}")

    engine = sym_cfg.get("engine") or RUNTIME.get("engine")
    managed = engine in {"band_trend", "pullback_swing", "council_v1"}
    scope = {"band_trend": "band50_v1", "pullback_swing": "pullback100_v1", "council_v1": "council_v1"}.get(engine, "legacy")
    state_key = f"{symbol}:{scope}" if managed else symbol
    managed_qty = amount(0)
    if managed:
        in_position, managed_qty = managed_position(DECISION_STATE, state_key, RUNTIME["trading_cfg"].get("execution_mode", "shadow"))
        result = strategy_func(closed_df, force=force, managed_position=in_position)
    else:
        result = strategy_func(closed_df, force=force)

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

    used = isinstance(result, dict) and bool(result.get('stats', {}).get('gpt_used'))
    local_reason = result.get('local_view', {}).get('reason', '') if isinstance(result, dict) else ''
    gpt_status = '已回覆' if used else ('不可用／失敗' if 'unavailable' in local_reason.lower() else '未呼叫')
    status.update(health='評估完成',decision=decision or 'HOLD',gpt=gpt_status,
                  reason=reason or local_reason or '策略未提供理由',order='無新訂單')
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
        base, quote = assets_from_symbol(symbol)
        balances = portfolio_before["breakdown"]
        submitted_quantity = order_quantity
        if managed and RUNTIME["trading_cfg"].get("execution_mode", "shadow") == "live":
            if (decision == "BUY" and in_position) or (decision == "SELL" and not in_position):
                status['order']='未送單：與策略持倉狀態衝突'
                logger.warning(f"[{symbol}] Decision conflicts with managed position; no order")
                return
            desired = managed_qty if decision == "SELL" else amount(order_quantity)
            submitted_quantity = market_quantity(client.get_symbol_info(symbol), desired, closed_df["close"].iloc[-1])
            order_quantity = float(submitted_quantity)
        if decision == "SELL" and balances.get(base, {}).get("balance", 0) < order_quantity:
            status['order']='未送單：可用資產不足'
            logger.warning(f"[{symbol}] SELL skipped: insufficient available {base} for configured lot")
            return
        if decision == "BUY":
            quote_balance = balances.get(quote, {}).get("balance", 0)
            if quote_balance < order_quantity * float(closed_df["close"].iloc[-1]) * 1.01:
                status['order']='未送單：報價資產不足（含費用緩衝）'
                logger.warning(f"[{symbol}] BUY skipped: insufficient quote balance including cost buffer")
                return
        if RUNTIME["trading_cfg"].get("execution_mode", "shadow") == "shadow":
            status['order']='shadow 模式，不送真實訂單'
            would_trade = DECISION_STATE.record_shadow(state_key, decision, closed_df.index[-1])
            logger.info(f"[{symbol}] SHADOW_ORDER side={decision} qty={order_quantity} would_trade={would_trade} bar={closed_df.index[-1]}")
            return
        permitted, guard_reason = DECISION_STATE.begin_order(state_key, decision, closed_df.index[-1])
        if not permitted:
            status['order']=f'未送單：{guard_reason}'
            logger.warning(f"[{symbol}] Order suppressed: {guard_reason}")
            return
        status['order']='送單中／尚未確認成交'
        order = execute_order(decision, symbol=symbol, quantity=submitted_quantity)
        if order:
            if managed:
                confirm_managed_order(DECISION_STATE, state_key, order, base, submitted_quantity, decision)
            else:
                DECISION_STATE.complete_order(state_key, order)
            status['order']=f'{decision} 已確認成交'
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
            status['order']='訂單失敗或結果不明，需核對交易所'
            logger.error(f"[{symbol}] Order failed or was not created: {decision}")
            send_message(f"[{symbol}] ⚠️ Order FAILED: {decision} (not recorded)")


_cycle_count = 0


def run_all_symbols(force: bool = False):
    global _cycle_count
    cycle_started = pd.Timestamp.now(tz='UTC')
    try:
        _maybe_reload_config()
    except Exception as exc:
        SYMBOL_STATUS['CONFIG']={'health':'設定錯誤','decision':'未評估','gpt':'未呼叫','reason':type(exc).__name__}
        _send_heartbeat(now=cycle_started)
        return
    SYMBOL_STATUS.pop('CONFIG',None)
    for sym_cfg in SYMBOL_RUNNERS:
        try:
            run_one_symbol(sym_cfg, force=force)
        except Exception as e:
            SYMBOL_STATUS.setdefault(sym_cfg['symbol'],{}).update(health='執行錯誤',reason=f'{type(e).__name__}；請檢查 bot log')
            logger.exception(f"[{sym_cfg['symbol']}] Error during strategy run: {e}")

    _send_heartbeat(now=cycle_started)
    # One snapshot per cycle (after all symbols processed).
    try:
        snap = get_portfolio_value(PORTFOLIO_ASSETS)
        record_snapshot(snap, label="cycle")
        _cycle_count += 1
        # Every 6 cycles (~1 hour at 10-min cadence), refresh the summary file and log it.
        if _cycle_count % max(1, 3600 // RUNTIME["poll_seconds"]) == 0:
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


def _send_heartbeat(now=None):
    try:
        symbols={s['symbol']:SYMBOL_STATUS.get(s['symbol'],{}) for s in SYMBOL_RUNNERS}
        if 'CONFIG' in SYMBOL_STATUS: symbols['CONFIG']=SYMBOL_STATUS['CONFIG']
        HEARTBEAT.send_if_due(send_message,symbols,RUNTIME['trading_cfg'].get('execution_mode','shadow'),now=now)
    except Exception as exc:
        logger.warning('Heartbeat unavailable ({})',type(exc).__name__)


if __name__ == "__main__":
    symbols_summary = ", ".join(f"{s['symbol']}(qty={s['order_quantity']},tf={s.get('timeframe',timeframe)},engine={s.get('engine') or RUNTIME.get('engine')})" for s in SYMBOL_RUNNERS)
    logger.info(f"Bot started | execution_mode={RUNTIME['trading_cfg'].get('execution_mode', 'shadow')} | strategy={strategy_name} | symbols=[{symbols_summary}] | timeframe={timeframe} | lookback={lookback}")
    try:
        initial_portfolio = get_portfolio_value(PORTFOLIO_ASSETS)
        logger.info(f"Initial portfolio | {_format_portfolio(initial_portfolio)}")
        send_message(f"🚀 Bot started | {_format_portfolio(initial_portfolio)}")
    except Exception as e:
        logger.warning(f"Could not fetch initial portfolio: {e}")
    logger.info("First forced run")
    run_all_symbols(force=True)

    while True:
        time.sleep(RUNTIME["poll_seconds"])
        run_all_symbols(force=False)
