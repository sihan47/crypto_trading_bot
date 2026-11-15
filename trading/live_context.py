"""Assemble live public context; failures remain explicit and never become quiet markets."""
from concurrent.futures import ThreadPoolExecutor
from trading.macro_calendar import fetch_calendar,utc
from trading.live_microstructure import fetch_telemetry


def get_live_context(symbol,news_context=None,now=None):
    now=utc(now)
    with ThreadPoolExecutor(max_workers=2) as pool:
        telemetry=pool.submit(fetch_telemetry,symbol,now)
        calendar=pool.submit(fetch_calendar,now)
        values=[]
        for name,future in [('telemetry',telemetry),('macro_calendar',calendar)]:
            try:values.append((name,future.result()))
            except Exception as exc:values.append((name,{'status':'unavailable','as_of':now.isoformat(),'error_type':type(exc).__name__}))
    return {**dict(values),'news':news_context or {'status':'unavailable'},
            'as_of':now.isoformat(),'max_size_multiplier':1.0,
            'sizing_policy':'Live data improves risk review; current BTC core profile remains capped at its configured base lot. No automatic 1.5x risk increase.'}
