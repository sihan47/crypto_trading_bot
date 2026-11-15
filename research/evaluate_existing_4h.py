"""Continuous full-history comparison; no optimization, GPT, or live orders."""
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


def main():
    config = yaml.safe_load((ROOT / 'config.yaml').read_text(encoding='utf-8'))
    ensemble = EnsembleParams(**config['strategy']['params']['ensemble'])
    specs = [
        ('sma', SMAParams(), run_sma_strategy, False),
        ('rsi', RSIParams(), run_rsi_strategy, False),
        ('macd', MACDParams(), run_macd_strategy, False),
        ('bollinger', BollingerParams(), run_bollinger_strategy, False),
        ('ema_adx', EMAADXParams(), run_ema_adx_strategy, True),
        ('donchian', DonchianParams(), run_donchian_strategy, True),
        ('mean_reversion', MeanReversionParams(), run_mean_reversion_strategy, False),
        ('zscore', ZScoreParams(), run_zscore_strategy, False),
    ]
    report = {'method': {
        'timeframe': '4h', 'fee_per_side': .001, 'slippage_per_side': .0005,
        'execution': 'continuous full history; next open; long-only; full allocation; final liquidation',
        'parameters': {name: asdict(p) for name, p, _, _ in specs},
        'ensemble_parameters': asdict(ensemble),
        'limitations': ['Default individual strategy parameters, not 15m optimized parameters.',
                        'Existing ensemble weights were selected using parts of this history; not blind validation.',
                        'No historical GPT/news replay; no fixed-lot sizing or live polling delay.',
                        'Only complete 4h groups retained; missing data not filled.',
                        'Each strategy warms up in cash; buy-and-hold enters at first available next open.'],
    }, 'symbols': {}}
    for symbol in ['BTCUSDT', 'BNBUSDT']:
        raw = pd.read_csv(ROOT / f'data/research_15m/{symbol}.csv', index_col=0, parse_dates=True)
        raw.index = pd.to_datetime(raw.index, utc=True)
        if raw.index.has_duplicates or not raw.index.is_monotonic_increasing:
            raise ValueError('Unique sorted input required')
        groups = raw.resample('4h', label='left', closed='left')
        df = groups.agg({'open': 'first', 'high': 'max', 'low': 'min', 'close': 'last', 'volume': 'sum'})
        df = df.loc[groups['close'].count().eq(16)]
        metrics = {}
        for name, params, fn, full in specs:
            result = fn(df if full else df['close'], params)
            metrics[name] = simulate(df, result['entries'], result['exits'])
        result = score_families(family_features(df, ensemble), ensemble)
        metrics['regime_v2'] = simulate(df, result['entries'], result['exits'])
        metrics['buy_hold'] = simulate(df, pd.Series(True, index=df.index), pd.Series(False, index=df.index))
        report['symbols'][symbol] = {
            'first_open': df.index[0].isoformat(), 'last_open': df.index[-1].isoformat(),
            'bars': len(df), 'gaps': int(df.index.to_series().diff().dropna().ne(pd.Timedelta(hours=4)).sum()),
            'results': metrics,
        }
        print(symbol, json.dumps(report['symbols'][symbol]), flush=True)
    path = ROOT / 'research/existing_4h_full_evaluation.json'
    path.write_text(json.dumps(report, indent=2, allow_nan=False), encoding='utf-8')
    print(path)


if __name__ == '__main__':
    main()
