"""
Run the PCA factor backtest from the command line.

Example:
  python -m scripts.run_backtest --data data/sample/IJR_small.txt --out reports/figures/cumret.png
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt

from src.utils.io import load_price_table
from src.factor.pca_strategy import PCAStrategyConfig, generate_positions_pca_trend
from src.backtest.engine import backtest_positions


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data", required=True, help="Path to price table (txt/csv). Must include a Date column.")
    p.add_argument("--lookback", type=int, default=252)
    p.add_argument("--num-factors", type=int, default=5)
    p.add_argument("--top-n", type=int, default=50)
    p.add_argument("--out", default="", help="Optional path to save cumret plot (png).")
    args = p.parse_args()

    prices = load_price_table(args.data)
    cfg = PCAStrategyConfig(lookback=args.lookback, num_factors=args.num_factors, top_n=args.top_n)
    positions = generate_positions_pca_trend(prices, cfg)

    res = backtest_positions(prices, positions)

    print(f"Annualized Return: {res.ann_return:.4f}")
    print(f"Annualized Vol:    {res.ann_vol:.4f}")
    print(f"Sharpe:            {res.sharpe:.4f}")
    print(f"Max Drawdown:      {res.max_drawdown:.4f}")
    print(f"Max DD Duration:   {int(res.max_drawdown_duration)}")

    plt.figure()
    plt.plot(res.cum_returns.index, res.cum_returns.values)
    plt.title("Cumulative Compounded Return")
    plt.xlabel("Date")
    plt.ylabel("Cumulative Return")

    if args.out:
        out = Path(args.out)
        out.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(out, dpi=200, bbox_inches="tight")
    else:
        plt.show()


if __name__ == "__main__":
    main()
