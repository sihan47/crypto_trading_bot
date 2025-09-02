import pandas as pd
import matplotlib.pyplot as plt
import os

SIGNALS_PATH = "artifacts/signals/BTCUSDT_signals.parquet"

def plot_signals(path=SIGNALS_PATH, start=None, end=None, strategies=None):
    df = pd.read_parquet(path)
    if start:
        df = df[df.index >= pd.to_datetime(start)]
    if end:
        df = df[df.index <= pd.to_datetime(end)]

    if strategies:
        cols = ["close"] + [c for c in df.columns if c in strategies]
    else:
        cols = df.columns

    # Plot price
    plt.figure(figsize=(14, 8))
    ax1 = plt.subplot(2, 1, 1)
    df["close"].plot(ax=ax1, color="black", label="Price")
    ax1.set_title("Price")
    ax1.legend()

    # Plot signals
    ax2 = plt.subplot(2, 1, 2)
    for col in df.columns:
        if col == "close":
            continue
        ax2.plot(df.index, df[col], label=col, alpha=0.7)
    ax2.set_title("Strategy Signals (-1=SELL, 0=HOLD, 1=BUY)")
    ax2.legend()

    plt.tight_layout()
    out_path = os.path.join("research", "signals_plot.png")
    plt.savefig(out_path)
    print(f"✅ Plot saved: {out_path}")
    plt.show()

if __name__ == "__main__":
    plot_signals(start="2024-08-01", end="2024-08-07")
