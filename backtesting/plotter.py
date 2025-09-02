import matplotlib.pyplot as plt
import pandas as pd
from pathlib import Path


def plot_equity_and_drawdown(equity: pd.Series, out_path: Path, title: str = "Equity Curve"):
    """
    Plot equity curve and drawdown chart.
    """
    dd = equity / equity.cummax() - 1

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8), sharex=True,
                                   gridspec_kw={'height_ratios': [3, 1]})

    # Equity curve
    ax1.plot(equity.index, equity.values, label="Equity")
    ax1.set_title(title)
    ax1.set_ylabel("Equity")
    ax1.legend()

    # Drawdown
    ax2.fill_between(dd.index, dd.values, 0, color="red", alpha=0.4)
    ax2.set_title("Drawdown")
    ax2.set_ylabel("Drawdown")
    ax2.set_xlabel("Date")

    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)
