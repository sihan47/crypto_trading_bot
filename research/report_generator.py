import json
from pathlib import Path
import pandas as pd

REPORT_DIR = Path(__file__).resolve().parent / "signals"


def load_results():
    results = []
    for file in REPORT_DIR.glob("*_full_signals.json"):
        with open(file, "r") as f:
            data = json.load(f)

        # filename format: BTCUSDT_{timeframe}_full_signals.json
        parts = file.stem.split("_")
        symbol, timeframe = parts[0], parts[1]

        for strat, content in data.items():
            stats = content.get("stats", {})
            results.append({
                "symbol": symbol,
                "timeframe": timeframe,
                "strategy": strat,
                "total_return": stats.get("total_return"),
                "sharpe": stats.get("sharpe"),
                "max_dd": stats.get("max_dd"),
                "win_rate": stats.get("win_rate"),
                "profit_factor": stats.get("profit_factor"),
                "trades": stats.get("trades"),
                "decisions_count": stats.get("decisions_count"),  # ✅ for GPT
            })
    return pd.DataFrame(results)


def generate_report(sort_by="total_return"):  # ✅ default changed
    df = load_results()
    if df.empty:
        print("⚠️ No results found. Run signal_runner first.")
        return

    # Sort by chosen metric
    if sort_by in df.columns:
        df = df.sort_values(by=sort_by, ascending=False)

    # Save CSV
    csv_path = REPORT_DIR / "summary_report.csv"
    df.to_csv(csv_path, index=False)
    print(f"✅ Summary CSV saved: {csv_path}")

    # Print markdown-like table (fallback to to_string if tabulate missing)
    try:
        print("\n📊 Performance Summary:")
        print(df.to_markdown(index=False))
    except Exception:
        print("\n📊 Performance Summary:")
        print(df.to_string(index=False))


if __name__ == "__main__":
    # Default now sorts by total_return
    generate_report(sort_by="total_return")
