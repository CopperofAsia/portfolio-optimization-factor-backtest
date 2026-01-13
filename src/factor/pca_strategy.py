"""
PCA-based factor strategy (trend-following flavor) from the original notebook.

Idea (daily re-fit):
1) Use past lookback window returns R (assets x days)
2) Extract PCA factors on R.T (days x assets) -> factor time series X (days x k)
3) Regress asset returns on factors (multi-output linear regression)
4) Predict returns in the lookback window and sum predictions per asset as a "cumulative expected return"
5) Long topN assets, short bottomN assets (equal-weight within each side)

This module only generates positions; the backtest engine computes PnL & metrics.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np
import pandas as pd
import statsmodels.api as sm
from sklearn.decomposition import PCA
from sklearn.linear_model import LinearRegression
from sklearn.multioutput import MultiOutputRegressor


@dataclass
class PCAStrategyConfig:
    lookback: int = 252
    num_factors: int = 5
    top_n: int = 50
    n_jobs: int = 4


def generate_positions_pca_trend(
    prices: pd.DataFrame,
    cfg: PCAStrategyConfig = PCAStrategyConfig(),
) -> pd.DataFrame:
    """
    Generate a positions table (same shape as prices) with values in {-1, 0, +1}.

    Parameters
    ----------
    prices : DataFrame
        Wide price table indexed by date, columns are tickers.
        Missing values are forward-filled before returns are computed.
    cfg : PCAStrategyConfig
        Strategy hyperparameters.

    Returns
    -------
    positions : DataFrame
        Positions by date and ticker. Positions at date t are the holdings decided at t.
        PnL should use positions.shift(1) * returns to avoid look-ahead.
    """
    if prices.shape[0] <= cfg.lookback + 1:
        raise ValueError("Not enough rows for the given lookback window.")

    df = prices.copy()
    df = df.sort_index()
    df = df.ffill()

    daily_ret = df.pct_change()
    positions = pd.DataFrame(0.0, index=df.index, columns=df.columns)

    end_index = df.shape[0]
    for t in range(cfg.lookback + 1, end_index):
        R = daily_ret.iloc[t - cfg.lookback : t, :].T  # assets x days

        # keep only assets with complete data in the window
        has_data = np.where(R.notna().all(axis=1))[0]
        R = R.dropna(axis=0, how="any")
        if R.shape[0] < 2:
            continue

        # PCA on days x assets
        pca = PCA()
        X = pca.fit_transform(R.T)[:, : cfg.num_factors]  # days x k
        X = sm.add_constant(X)  # add intercept column
        Y = R.T  # days x assets

        clf = MultiOutputRegressor(
            LinearRegression(fit_intercept=False),
            n_jobs=cfg.n_jobs,
        ).fit(X, Y)

        cum_exp_ret = np.sum(clf.predict(X), axis=0)  # per asset
        idx_sort = np.argsort(cum_exp_ret)

        # map back to original column indices
        short_idx = has_data[idx_sort[: cfg.top_n]]
        long_idx = has_data[idx_sort[-cfg.top_n :]]

        positions.iloc[t, short_idx] = -1.0
        positions.iloc[t, long_idx] = 1.0

    return positions
