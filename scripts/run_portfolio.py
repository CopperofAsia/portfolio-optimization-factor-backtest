"""
Run a simple Markowitz + Black–Litterman demo from the command line.

Example:
  python -m scripts.run_portfolio --data data/sample/prices.csv

Input format:
- txt/csv file with a Date column and one column per asset price.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

from src.utils.io import load_price_table
from src.portfolio.markowitz import efficient_frontier, max_sharpe, portfolio_return, portfolio_volatility
from src.portfolio.black_litterman import implied_equilibrium_returns_pi, black_litterman_posterior


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data", required=True, help="Path to price table (txt/csv) with Date column.")
    p.add_argument("--rf", type=float, default=0.0, help="Annualized risk-free rate (default 0).")
    args = p.parse_args()

    prices = load_price_table(args.data)
    ret = prices.pct_change().dropna()

    mu_daily = ret.mean().values
    cov_daily = ret.cov().values

    # annualize
    mu = mu_daily * 252
    cov = cov_daily * 252

    # Markowitz efficient frontier
    targets, vols, ws = efficient_frontier(mu, cov, n_points=30, long_only=True)

    w_tan = max_sharpe(mu, cov, rf=args.rf, long_only=True)
    tan_r = portfolio_return(w_tan, mu)
    tan_vol = portfolio_volatility(w_tan, cov)

    plt.figure()
    plt.plot(vols, targets)
    plt.scatter([tan_vol], [tan_r])
    plt.xlabel("Volatility")
    plt.ylabel("Expected Return")
    plt.title("Efficient Frontier (Markowitz)")
    plt.show()

    # --- Black–Litterman demo (toy views) ---
    n = len(mu)
    w_mkt = np.repeat(1.0 / n, n)  # placeholder if you don't have market caps
    delta = 2.5  # placeholder
    pi = implied_equilibrium_returns_pi(delta, cov, w_mkt)

    # Example: one view: asset 0 will outperform asset 1 by 2% annualized
    P = np.zeros((1, n))
    P[0, 0] = 1
    P[0, 1] = -1
    q = np.array([0.02])
    Omega = np.array([[0.05 ** 2]])

    mu_bl, cov_bl = black_litterman_posterior(cov=cov, pi=pi, P=P, q=q, Omega=Omega, tau=0.05)

    print("Tangency weights (Markowitz):")
    print(dict(zip(prices.columns, np.round(w_tan, 4))))
    print("\nPosterior mean (Black–Litterman, demo view):")
    print(dict(zip(prices.columns, np.round(mu_bl, 4))))


if __name__ == "__main__":
    main()
