"""Predeclared small weight comparison; final holdout is never used for selection.

2023-24 development folds; 2025 validation for selection; May-Nov 2026 holdout.
Historical LLM/news decisions cannot be reconstructed and are NOT simulated.
"""
import json
import argparse
import sys
from dataclasses import asdict, replace
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import pandas as pd

from backtesting.next_bar import simulate
from strategies.regime_ensemble import EnsembleParams, family_features, score_families
from strategies.sma_strategy import SMAParams, run_sma_strategy
from strategies.rsi_strategy import RSIParams, run_rsi_strategy
from strategies.macd_strategy import MACDParams, run_macd_strategy
from strategies.bollinger_strategy import BollingerParams, run_bollinger_strategy
from strategies.ema_adx_strategy import EMAADXParams, run_ema_adx_strategy
from strategies.donchian_strategy import DonchianParams, run_donchian_strategy
from strategies.mean_reversion_strategy import MeanReversionParams, run_mean_reversion_strategy
from strategies.zscore_strategy import ZScoreParams, run_zscore_strategy

OUT = ROOT / 'research' / 'ensemble_evaluation.json'
FOLDS = {
    'development_2023': ('2023-01-01', '2024-01-01'),
    'development_2024': ('2024-01-01', '2025-01-01'),
    'validation_2025': ('2025-01-01', '2026-01-01'),
    'holdout_2026': ('2026-05-02', '2026-11-14'),
}
# Fixed BEFORE examining results: no large parameter search.
CANDIDATES = {
    'balanced': EnsembleParams(.4, .3, .3),
    'trend_focus': EnsembleParams(.6, .3, .1),
    'breakout_focus': EnsembleParams(.3, .6, .1),
    'range_focus': EnsembleParams(.2, .2, .6),
}


def baseline_signals(df, best, symbol):
    result = {}
    specs = [
        ('sma', SMAParams, run_sma_strategy, False),
        ('rsi', RSIParams, run_rsi_strategy, False),
        ('macd', MACDParams, run_macd_strategy, False),
        ('bollinger', BollingerParams, run_bollinger_strategy, False),
        ('ema_adx', EMAADXParams, run_ema_adx_strategy, True),
        ('donchian', DonchianParams, run_donchian_strategy, True),
        ('mean_reversion', MeanReversionParams, run_mean_reversion_strategy, False),
        ('zscore', ZScoreParams, run_zscore_strategy, False),
    ]
    for name, cls, fn, full in specs:
        params = cls(**best.get(f'{symbol}_15m_{name}', {}))
        output = fn(df if full else df['close'], params)
        result['existing_' + name] = (output['entries'], output['exits'])
    result['buy_hold'] = (pd.Series(True, index=df.index), pd.Series(False, index=df.index))
    result['cash'] = (pd.Series(False, index=df.index), pd.Series(False, index=df.index))
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--timeframe', choices=['15m', '1h', '4h'], default='15m')
    args = parser.parse_args()
    timeframe = args.timeframe
    output_path = OUT if timeframe == '15m' else OUT.with_name('ensemble_evaluation_' + timeframe + '.json')
    best = json.loads((ROOT / 'research/best_params.json').read_text(encoding='utf-8'))
    report = {'method': {
        'timeframe': timeframe, 'fee_per_side': .001, 'slippage_per_side': .0005,
        'execution': 'next open; long-only single full allocation; force-close at final bar',
        'selection': 'highest mean return + 0.5 * negative drawdown across 2023, 2024, 2025 and both assets',
        'caveats': ['No historical GPT/news replay; results are deterministic strategy component only.',
                    'Existing best parameters were tuned through 2025/May 2026; only final holdout is fair for those baselines.',
                    'Full allocation research model differs from fixed-quantity live orders.',
                    'Four candidate weights tested; no claim of statistical significance.'],
        'folds': FOLDS, 'candidates': {k: asdict(v) for k,v in CANDIDATES.items()},
    }, 'symbols': {}}
    cached = {}
    # Compute validation metrics only; selection is frozen before evaluating holdout.
    for symbol in ['BTCUSDT', 'BNBUSDT']:
        df = pd.read_csv(ROOT / f'data/research_15m/{symbol}.csv', index_col=0, parse_dates=True)
        df.index = pd.to_datetime(df.index, utc=True)
        if timeframe != '15m':
            groups = df.resample(timeframe, label='left', closed='left')
            counts = groups['close'].count()
            df = groups.agg({'open':'first','high':'max','low':'min','close':'last','volume':'sum'})
            df = df.loc[counts == (4 if timeframe == '1h' else 16)]
        features = family_features(df)
        signals = {name:(s['entries'],s['exits']) for name,p in CANDIDATES.items()
                   for s in [score_families(features,p)]}
        signals.update(baseline_signals(df,best if timeframe == '15m' else {},symbol))
        cached[symbol] = (df, features, signals)
        sr = {'bars':len(df), 'start':str(df.index[0]), 'end':str(df.index[-1]), 'folds':{}}
        for fold,(start,end) in FOLDS.items():
            if fold.startswith('holdout'):
                continue
            block = df.loc[(df.index >= start) & (df.index < end)]
            sr['folds'][fold] = {name:simulate(block, a.loc[block.index],b.loc[block.index]) for name,(a,b) in signals.items()}
        report['symbols'][symbol] = sr
    scores = {}
    for name in CANDIDATES:
        metrics = [s['folds'][fold][name] for s in report['symbols'].values()
                   for fold in FOLDS if not fold.startswith('holdout')]
        scores[name] = sum(m['return_pct'] + .5*m['max_drawdown_pct'] for m in metrics)/len(metrics)
    selected = max(scores, key=scores.get)
    report['selection_scores'] = scores
    report['selected'] = selected
    report['selected_params'] = asdict(CANDIDATES[selected])
    for symbol,(df,features,signals) in cached.items():
        start,end = FOLDS['holdout_2026']
        block = df.loc[(df.index >= start) & (df.index < end)]
        # Report every predeclared candidate, without changing the frozen winner.
        report['symbols'][symbol]['folds']['holdout_2026'] = {
            name:simulate(block,a.loc[block.index],b.loc[block.index]) for name,(a,b) in signals.items()}
        a,b = signals[selected]
        report['symbols'][symbol]['holdout_stress_costs'] = simulate(block,a.loc[block.index],b.loc[block.index],fee=.002,slippage=.001)
    output_path.write_text(json.dumps(report,indent=2,allow_nan=False),encoding='utf-8')
    print('Selection (before holdout):', selected, scores)
    for symbol,s in report['symbols'].items():
        print(symbol)
        for name,m in s['folds']['holdout_2026'].items():
            print(name, {k:round(v,2) for k,v in m.items() if k in ['return_pct','max_drawdown_pct','round_trips','exposure_pct']})
    print('Report:', output_path)


if __name__ == '__main__':
    main()
