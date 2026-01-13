"""
Simple vectorized backtest engine for positions tables.

Assumptions (matching the original notebook):
- Positions are -1/0/+1 signals (not dollar weights).
- Capital is proportional to sum(|positions|) each day.
- Daily portfolio return is:
    ret_t = sum_i pos_{t-1,i} * r_{t,i} / capital_t
  where capital_t = sum_i |pos_{t-1,i}|
- If capital_t == 0, ret_t is set to 0.

This is intentionally simple and educational.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import numpy as np
import pandas as pd

from .performance import (
    cumulative_compounded_returns,
    sharpe_ratio,
    calculateMaxDD,
    annualized_return,
    annualized_volatility,
)


@dataclass
class BacktestResult:
    daily_returns: pd.Series
    cum_returns: pd.Series
    ann_return: float
    ann_vol: float
    sharpe: float
    max_drawdown: float
    max_drawdown_duration: float
    max_dd_index: int


def backtest_positions(prices: pd.DataFrame, positions: pd.DataFrame, periods_per_year: int = 252) -> BacktestResult:
    """
    Backtest a positions table against a price panel.

    Parameters
    ----------
    prices : DataFrame
        Price table indexed by date, columns tickers.
    positions : DataFrame
        Same shape as prices with positions at each date.

    Returns
    -------
    BacktestResult
    """
    if not prices.index.equals(positions.index) or not prices.columns.equals(positions.columns):
        raise ValueError("prices and positions must have the same index and columns.")

    prices = prices.sort_index().ffill()
    daily_ret = prices.pct_change()

    pos_lag = positions.shift(1).fillna(0.0)

    capital = pos_lag.abs().sum(axis=1)
    capital_safe = capital.copy()
    capital_safe[capital_safe == 0] = 1.0

    port_ret = (pos_lag * daily_ret).sum(axis=1) / capital_safe
    port_ret[capital == 0] = 0.0
    port_ret = port_ret.fillna(0.0)

    cum = pd.Series(cumulative_compounded_returns(port_ret.values), index=port_ret.index, name="cum_return")

    ann_r = annualized_return(port_ret.values, periods_per_year=periods_per_year)
    ann_vol = annualized_volatility(port_ret.values, periods_per_year=periods_per_year)
    sr = sharpe_ratio(port_ret.values, rf=0.0, periods_per_year=periods_per_year)

    maxDD, maxDDD, i = calculateMaxDD(cum.values)

    return BacktestResult(
        daily_returns=port_ret.rename("daily_return"),
        cum_returns=cum,
        ann_return=ann_r,
        ann_vol=ann_vol,
        sharpe=sr,
        max_drawdown=maxDD,
        max_drawdown_duration=maxDDD,
        max_dd_index=i,
    )
