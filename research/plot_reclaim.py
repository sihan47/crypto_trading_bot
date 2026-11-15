"""Plot the research equity curve and realized trade distribution."""
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]


def main():
    folder = ROOT/'output/structural_reclaim'
    eq = pd.read_csv(folder/'holdout_equity.csv', index_col=0).iloc[:, 0]
    eq.index = pd.to_datetime(eq.index, utc=True)
    trades = pd.read_csv(folder/'holdout_trades.csv')
    fig, axes = plt.subplots(3, 1, figsize=(11, 10), constrained_layout=True)
    axes[0].plot(eq.index, (eq/10000-1)*100)
    axes[0].set(ylabel='Net return (%)', title='G: structural reclaim — exposed-history holdout (2025 onward)')
    axes[1].fill_between(eq.index, (eq/eq.cummax().clip(lower=10000)-1)*100, 0, color='firebrick', alpha=.5)
    axes[1].set(ylabel='Drawdown (%)')
    pnl = trades.net_pnl
    axes[2].bar(range(1, len(pnl)+1), pnl, color=['seagreen' if x>0 else 'firebrick' for x in pnl])
    axes[2].set(xlabel='Trade number', ylabel='Realized net PnL (USDT)', title='10,000 USDT initial capital; 10 bp fee + 5 bp slippage per side')
    for ax in axes:
        ax.grid(alpha=.2)
    fig.savefig(folder/'review.png', dpi=160)
    plt.close(fig)


if __name__ == '__main__': main()
