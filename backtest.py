import argparse
from backtesting.backtester import run_backtest

def main():
    parser = argparse.ArgumentParser(description="Crypto trading backtest runner")
    parser.add_argument("--symbol", type=str, required=True, help="Trading pair (e.g. BTCUSDT)")
    parser.add_argument("--start", type=str, help="Backtest start date (YYYY-MM-DD)")
    parser.add_argument("--end", type=str, help="Backtest end date (YYYY-MM-DD)")
    parser.add_argument("--strategy", type=str, default="sma", help="Strategy name (sma/gpt/...)")
    parser.add_argument("--verbose", action="store_true", help="Print debug info")
    parser.add_argument("--plot", action="store_true", help="Show plot interactively")

    args = parser.parse_args()

    results = run_backtest(
        symbol=args.symbol,
        strategy_name=args.strategy,
        start_date=args.start,
        end_date=args.end,
        verbose=args.verbose,
        plot=args.plot
    )

    print("\nBacktest Results:")
    for k, v in results.items():
        print(f"{k}: {v}")

if __name__ == "__main__":
    main()
