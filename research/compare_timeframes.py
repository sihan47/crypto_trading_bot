"""Existing strategies across 15m through 4h, research only."""
import json
import sys
from dataclasses import asdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import pandas as pd
import yaml
from backtesting.next_bar import simulate
from research.evaluate_ensemble import (
    SMAParams, run_sma_strategy, RSIParams, run_rsi_strategy,
    MACDParams, run_macd_strategy, BollingerParams, run_bollinger_strategy,
    EMAADXParams, run_ema_adx_strategy, DonchianParams, run_donchian_strategy,
    MeanReversionParams, run_mean_reversion_strategy, ZScoreParams, run_zscore_strategy,
)
from strategies.regime_ensemble import EnsembleParams, family_features, score_families


SPECS = [
    ('sma', SMAParams(), run_sma_strategy, False),
    ('rsi', RSIParams(), run_rsi_strategy, False),
    ('macd', MACDParams(), run_macd_strategy, False),
    ('bollinger', BollingerParams(), run_bollinger_strategy, False),
    ('ema_adx', EMAADXParams(), run_ema_adx_strategy, True),
    ('donchian', DonchianParams(), run_donchian_strategy, True),
    ('mean_reversion', MeanReversionParams(), run_mean_reversion_strategy, False),
    ('zscore', ZScoreParams(), run_zscore_strategy, False),
]

TIMEFRAMES = {'15m': ('15min', 1), '30m': ('30min', 2), '1h': ('1h', 4), '2h': ('2h', 8), '4h': ('4h', 16)}


def main():
    config = yaml.safe_load((ROOT / 'config.yaml').read_text(encoding='utf-8'))
    ensemble = EnsembleParams(**config['strategy']['params']['ensemble'])
    report = {'method': {'fee_per_side': .001, 'slippage_per_side': .0005,
        'parameters': {n: asdict(p) for n,p,_,_ in SPECS}, 'ensemble': asdict(ensemble),
        'selection': 'Report historical return and daily Sharpe leaders separately; no optimization of parameters.',
        'limitations': ['Same bar counts at every timeframe imply different elapsed lookbacks.',
            'Full allocation, next open, long only; no GPT/news replay or actual fixed lots.',
            'Historical selection across 45 candidates per asset is exploratory, not blind validation.',
            'Annual slices start in cash and force liquidation; indicators use prior history.',
            'Full run is continuous; annual returns cannot be added to reproduce it.']}, 'symbols': {}}
    for symbol in ['BTCUSDT', 'BNBUSDT']:
        raw = pd.read_csv(ROOT / f'data/research_15m/{symbol}.csv', index_col=0, parse_dates=True)
        raw.index = pd.to_datetime(raw.index, utc=True)
        assert raw.index.is_monotonic_increasing and not raw.index.has_duplicates
        report['symbols'][symbol] = {}
        for tf,(rule,count) in TIMEFRAMES.items():
            groups = raw.resample(rule, label='left', closed='left')
            df = groups.agg({'open':'first','high':'max','low':'min','close':'last','volume':'sum'})
            df = df.loc[groups['close'].count().eq(count)]
            signals = {}
            for name,params,fn,full in SPECS:
                r = fn(df if full else df['close'], params)
                signals[name] = (r['entries'],r['exits'])
            r = score_families(family_features(df,ensemble),ensemble)
            signals['regime_v2'] = (r['entries'],r['exits'])
            signals['buy_hold'] = (pd.Series(True,index=df.index),pd.Series(False,index=df.index))
            results = {}
            for name,(a,b) in signals.items():
                full = simulate(df,a,b)
                years = {}
                for year in sorted(set(df.index.year)):
                    block = df.loc[df.index.year == year]
                    years[str(year)] = simulate(block,a.loc[block.index],b.loc[block.index])
                results[name] = {'full':full,'years':years}
            report['symbols'][symbol][tf] = {'first_open':df.index[0].isoformat(),
                'last_open':df.index[-1].isoformat(),'bars':len(df),'results':results}
            path = ROOT / 'research/timeframe_comparison.json'
            path.write_text(json.dumps(report,indent=2,allow_nan=False),encoding='utf-8')
            winner = max((n for n in results if n != 'buy_hold'),key=lambda n:results[n]['full']['return_pct'])
            print(symbol,tf,winner,round(results[winner]['full']['return_pct'],2),flush=True)
    print('Completed:',path)


if __name__ == '__main__':
    main()
