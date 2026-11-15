"""Render the completed timeframe comparison as a reviewable Markdown report."""
import json
from pathlib import Path

ROOT=Path(__file__).resolve().parent

def main():
    report=json.loads((ROOT/'timeframe_comparison.json').read_text(encoding='utf-8'))
    frames=['15m','30m','1h','2h','4h']
    assert all(list(s)==frames for s in report['symbols'].values()), 'Comparison is incomplete'
    lines=['# Timeframe comparison: 15m to 4h','',
        'Approximately September 2022 through November 13, 2026 UTC. Each asset is simulated independently. Same strategy bar-count parameters across timeframes (different elapsed lookbacks); no tuning in this run. Nine deterministic strategies per timeframe, plus buy-and-hold. GPT and historical news are excluded.', '',
        'Full allocation, long only, continuous capital, next-open fills; fees 0.10% and slippage 0.05% on each side. Each indicator warms up in cash. Complete candles only, missing bars not filled. Annual slices reset to cash with prior indicator history; they do not compound to the full-period result. 2022 and 2026 are partial years.', '',
        'Historical rankings across 45 candidates per asset are exploratory, not independent validation. Existing ensemble weights were previously chosen using parts of the same history. Returns are cumulative, not annualized. Daily Sharpe uses 365 days and zero risk-free rate. These are not account returns or a test of the complete GPT bot.', '',
        '## Highest-return strategy at each timeframe','',
        '| Timeframe | BTC strategy | BTC return | BTC max drawdown | BNB strategy | BNB return | BNB max drawdown |',
        '| --- | --- | ---: | ---: | --- | ---: | ---: |']
    candidates={}
    for symbol,tfs in report['symbols'].items():
        candidates[symbol]=[(tf,n,v) for tf,d in tfs.items() for n,v in d['results'].items() if n!='buy_hold']
    for tf in frames:
        cells=[]
        for symbol in report['symbols']:
            _,n,v=max((c for c in candidates[symbol] if c[0]==tf),key=lambda c:c[2]['full']['return_pct'])
            m=v['full'];cells += [n,f"{m['return_pct']:+.2f}%",f"{m['max_drawdown_pct']:.2f}%"]
        lines.append('| '+' | '.join([tf]+cells)+' |')
    lines += ['', '## Overall leaders and annual sensitivity', '']
    for symbol,cs in candidates.items():
        for label,key in [('Highest return','return_pct'),('Highest daily Sharpe','sharpe_daily')]:
            tf,n,v=max(cs,key=lambda c:c[2]['full'][key]); m=v['full']
            lines += [f"### {symbol}: {label} ? {tf} {n}",'',f"Return {m['return_pct']:+.2f}%; max drawdown {m['max_drawdown_pct']:.2f}%; daily Sharpe {m['sharpe_daily']:.3f}; round trips {m['round_trips']}.",'', '| Year | Return | Max drawdown | Round trips |','| --- | ---: | ---: | ---: |']
            for y,a in v['years'].items():
                lines.append(f"| {y} | {a['return_pct']:+.2f}% | {a['max_drawdown_pct']:.2f}% | {a['round_trips']} |")
            lines.append('')
    lines += ['## All results','', '| Symbol | Timeframe | Strategy | Return | Max drawdown | Daily Sharpe | Round trips |','| --- | --- | --- | ---: | ---: | ---: | ---: |']
    for symbol,tfs in report['symbols'].items():
        for tf,d in tfs.items():
            for n,v in d['results'].items():
                m=v['full'];lines.append(f"| {symbol} | {tf} | {n} | {m['return_pct']:+.2f}% | {m['max_drawdown_pct']:.2f}% | {m['sharpe_daily']:.3f} | {m['round_trips']} |")
    lines += ['', 'Reproduce: `python research/compare_timeframes.py`, then `python research/summarize_timeframes.py`. No live configuration or VPS changes.']
    (ROOT/'TIMEFRAME_REVIEW.md').write_text(chr(10).join(lines)+chr(10),encoding='utf-8')

if __name__=='__main__':
    main()
