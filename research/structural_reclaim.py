"""Research G: delayed structural reclaim after a deleveraging event."""
from dataclasses import dataclass
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import json
import numpy as np
import pandas as pd
from research.structural_btc import Parameters, features, run_research, simulate, metrics
from research.structural_deleveraging import read_panel
from research.structural_short_covering import covering_signals, CoverParameters
ROOT = Path(__file__).resolve().parents[1]


@dataclass(frozen=True)
class ReclaimParameters:
    day: int = 24
    week: int = 168
    shock: float = 2.
    recovery_hours: int = 24
    trail: float = 2.
    max_hours: int = 168
    risk: float = .005
    liquidity_cap: float = .001


def reclaim_signals(panel, p=ReclaimParameters(), extra_delay_hours=0):
    event = covering_signals(panel, CoverParameters(week=p.week))
    end = (panel.create_time + pd.Timedelta(minutes=5)).dt.ceil('1h')
    event['reclaim'] = panel.spot_open.reindex(pd.DatetimeIndex(end)).to_numpy()
    event['event_end'] = end
    event['candidate'] = (event.oi_z <= -p.shock) & (event.price_z <= -p.shock)
    event.index += pd.Timedelta(hours=extra_delay_hours)
    event = event.reindex(panel.index)
    result = pd.DataFrame({'signal': False, 'initial_stop': np.nan}, index=panel.index)
    active = None
    previous = None
    for stamp, row in panel.iterrows():
        if previous is not None and stamp-previous != pd.Timedelta(hours=1):
            active = None
        previous = stamp
        e = event.loc[stamp]
        if row.spot_tradeable != True or not np.isfinite(row.spot_close) or not np.isfinite(row.spot_low):
            active = None
            continue
        # Never replace a pending event with a later event's more convenient low.
        if active is None and e.candidate == True:
            start = e.event_end
            history = panel.loc[(panel.index > start) & (panel.index <= stamp)]
            valid = (pd.notna(start) and start <= stamp and np.isfinite(e.initial_stop)
                     and np.isfinite(e.reclaim) and e.reclaim > e.initial_stop
                     and len(history) == int((stamp-start)/pd.Timedelta(hours=1))
                     and history.spot_tradeable.all()
                     and history.spot_low.notna().all()
                     and not history.spot_low.le(e.initial_stop).any())
            if valid:
                active = (start, e.initial_stop, e.reclaim)
        if active is None:
            continue
        start, stop, reclaim = active
        if stamp-start > pd.Timedelta(hours=p.recovery_hours) or (stamp > start and row.spot_low <= stop):
            active = None
        elif stamp > start and row.spot_close >= reclaim:
            result.loc[stamp] = [True, stop]
            active = None
    return result


def reclaim_features(raw, p=ReclaimParameters(), panel=None, extra_delay_hours=0):
    panel = read_panel() if panel is None else panel
    signals = reclaim_signals(panel, p, extra_delay_hours)
    signals.index -= pd.Timedelta(minutes=15)
    f = features(raw, Parameters(day=p.day, week=p.week, trail=p.trail,
                                max_hours=p.max_hours, risk=p.risk, liquidity_cap=p.liquidity_cap))
    evidence = signals.reindex(f.index)
    f['initial_stop'] = evidence.initial_stop
    f['signal'] = evidence.signal.eq(True) & f.sigma_day.notna() & f.shock.notna()
    return f


def main():
    panel = read_panel(); p = ReclaimParameters()
    report = run_research('reclaim', lambda raw, params: reclaim_features(raw, params, panel), p)
    raw = pd.read_csv(ROOT/'data/research_15m/BTCUSDT.csv', index_col=0)
    raw.index = pd.to_datetime(raw.index, utc=True)
    raw = raw.loc[raw.volume > 0]
    report['extra_data_delays'] = {}
    for hours in [1, 2]:
        f = reclaim_features(raw, p, panel, hours)
        eq, tr, cost = simulate(raw.loc[raw.index >= '2025-01-01'], f, p)
        report['extra_data_delays'][str(hours)] = metrics(eq, tr, cost)
    report['tests']['extra_data_delays_positive'] = all(v['return_pct'] > 0 for v in report['extra_data_delays'].values())
    report['passed'] = all(report['tests'].values())
    (ROOT/'research/structural_reclaim_evaluation.json').write_text(json.dumps(report, indent=2, allow_nan=False), encoding='utf-8')


if __name__ == '__main__':
    main()
