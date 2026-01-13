"""
Markowitz mean-variance portfolio optimization utilities.

This module focuses on:
- portfolio return / volatility
- global minimum variance portfolio
- max Sharpe portfolio (tangency)
- efficient frontier (min variance for target return)

The functions are deliberately written to work with plain numpy arrays
to keep the core optimization reusable.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple, Optional

import numpy as np
from scipy.optimize import minimize


def portfolio_return(w: np.ndarray, mu: np.ndarray) -> float:
    """Expected portfolio return."""
    return float(np.dot(w, mu))


def portfolio_volatility(w: np.ndarray, cov: np.ndarray) -> float:
    """Portfolio standard deviation."""
    return float(np.sqrt(np.dot(w, np.dot(cov, w))))


def _weight_bounds(n: int, long_only: bool) -> Tuple[Tuple[float, float], ...]:
    return tuple((0.0, 1.0) for _ in range(n)) if long_only else tuple((-1.0, 1.0) for _ in range(n))


def global_min_variance(cov: np.ndarray, long_only: bool = True) -> np.ndarray:
    """
    Compute the Global Minimum Variance (GMV) portfolio weights.

    Constraints:
      - sum(w) = 1
      - (optional) long-only bounds
    """
    n = cov.shape[0]
    x0 = np.repeat(1.0 / n, n)

    cons = ({"type": "eq", "fun": lambda w: np.sum(w) - 1.0},)
    bounds = _weight_bounds(n, long_only)

    def obj(w: np.ndarray) -> float:
        return np.dot(w, cov @ w)

    res = minimize(obj, x0, method="SLSQP", bounds=bounds, constraints=cons)
    if not res.success:
        raise RuntimeError(f"GMV optimization failed: {res.message}")
    return np.asarray(res.x)


def max_sharpe(mu: np.ndarray, cov: np.ndarray, rf: float = 0.0, long_only: bool = True) -> np.ndarray:
    """
    Compute the tangency portfolio weights by maximizing Sharpe ratio.

    Constraints:
      - sum(w) = 1
      - (optional) long-only bounds
    """
    n = cov.shape[0]
    x0 = np.repeat(1.0 / n, n)

    cons = ({"type": "eq", "fun": lambda w: np.sum(w) - 1.0},)
    bounds = _weight_bounds(n, long_only)

    def neg_sharpe(w: np.ndarray) -> float:
        r = portfolio_return(w, mu)
        vol = portfolio_volatility(w, cov)
        # numerical safety
        if vol <= 0:
            return 1e9
        return -((r - rf) / vol)

    res = minimize(neg_sharpe, x0, method="SLSQP", bounds=bounds, constraints=cons)
    if not res.success:
        raise RuntimeError(f"Max-Sharpe optimization failed: {res.message}")
    return np.asarray(res.x)


def min_variance_for_target_return(
    mu: np.ndarray,
    cov: np.ndarray,
    target_return: float,
    long_only: bool = True,
) -> np.ndarray:
    """
    Compute minimum-variance weights for a given target expected return.

    Constraints:
      - sum(w) = 1
      - w^T mu = target_return
      - (optional) long-only bounds
    """
    n = cov.shape[0]
    x0 = np.repeat(1.0 / n, n)

    cons = (
        {"type": "eq", "fun": lambda w: np.sum(w) - 1.0},
        {"type": "eq", "fun": lambda w: np.dot(w, mu) - float(target_return)},
    )
    bounds = _weight_bounds(n, long_only)

    def obj(w: np.ndarray) -> float:
        return np.dot(w, cov @ w)

    res = minimize(obj, x0, method="SLSQP", bounds=bounds, constraints=cons)
    if not res.success:
        raise RuntimeError(f"Efficient frontier optimization failed: {res.message}")
    return np.asarray(res.x)


def efficient_frontier(
    mu: np.ndarray,
    cov: np.ndarray,
    n_points: int = 30,
    long_only: bool = True,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Generate an efficient frontier by sweeping target returns.

    Returns
    -------
    target_returns : (n_points,) array
    vols          : (n_points,) array
    weights       : (n_points, n_assets) array
    """
    gmv_w = global_min_variance(cov, long_only=long_only)
    gmv_r = portfolio_return(gmv_w, mu)

    maxr_w = max_sharpe(mu, cov, rf=0.0, long_only=long_only)
    max_r = portfolio_return(maxr_w, mu)

    # In case of degenerate inputs, ensure proper ordering
    lo, hi = (min(gmv_r, max_r), max(gmv_r, max_r))
    targets = np.linspace(lo, hi, n_points)

    ws = []
    vols = []
    for tr in targets:
        w = min_variance_for_target_return(mu, cov, tr, long_only=long_only)
        ws.append(w)
        vols.append(portfolio_volatility(w, cov))

    return targets, np.asarray(vols), np.vstack(ws)
