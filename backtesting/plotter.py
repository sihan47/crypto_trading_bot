import matplotlib.pyplot as plt
import os

def plot_backtest(df, symbol, save=True, show=False, out_dir="logs"):
    plt.figure(figsize=(12, 6))
    plt.plot(df.index, df["equity"], label="Equity", color="blue", linewidth=1.5)
    plt.legend()
    plt.title(f"Equity Curve - {symbol}")
    plt.xlabel("Time")
    plt.ylabel("Equity (USD)")

    if save:
        os.makedirs(out_dir, exist_ok=True)
        out_path = os.path.join(out_dir, f"backtest_equity_{symbol}.png")
        plt.savefig(out_path)
        print(f"✅ Backtest equity curve saved: {out_path}")

    if show:
        plt.show(block=True)
    else:
        plt.close()
