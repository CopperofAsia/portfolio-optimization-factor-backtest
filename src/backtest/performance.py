"""
Backtest performance metrics.

Includes maximum drawdown (copied from the original project),
plus common helpers used in the notebooks:
- cumulative compounded return series
- annualized return / volatility
- Sharpe ratio
"""

from __future__ import annotations

from typing import Tuple
import numpy as np


def cumulative_compounded_returns(ret: np.ndarray) -> np.ndarray:
    """
    Convert daily simple returns into compounded cumulative returns:
      cumRet_t = Π_{i<=t} (1 + r_i) - 1
    """
    ret = np.asarray(ret, dtype=float)
    return np.cumprod(1.0 + ret) - 1.0


def annualized_return(ret: np.ndarray, periods_per_year: int = 252) -> float:
    ret = np.asarray(ret, dtype=float)
    return float(np.nanmean(ret) * periods_per_year)


def annualized_volatility(ret: np.ndarray, periods_per_year: int = 252) -> float:
    ret = np.asarray(ret, dtype=float)
    return float(np.nanstd(ret) * np.sqrt(periods_per_year))


def sharpe_ratio(ret: np.ndarray, rf: float = 0.0, periods_per_year: int = 252) -> float:
    """
    Sharpe ratio using simple daily returns and annualization.

    rf is annualized risk-free rate. For short educational projects, rf=0 is common.
    """
    r_ann = annualized_return(ret, periods_per_year=periods_per_year)
    vol_ann = annualized_volatility(ret, periods_per_year=periods_per_year)
    if vol_ann == 0:
        return float("nan")
    return float((r_ann - rf) / vol_ann)


def calculateMaxDD(cumRet: np.ndarray) -> Tuple[float, float, int]:
    """
    Calculation of maximum drawdown and maximum drawdown duration based on
    cumulative COMPOUNDED returns. cumRet must be a compounded cumulative return.
    i is the index of the day with maxDD.

    Returns
    -------
    maxDD  : most negative drawdown (<= 0)
    maxDDD : maximum drawdown duration (in periods)
    i      : index of maxDD
    """
    cumRet = np.asarray(cumRet, dtype=float)

    highwatermark = np.zeros(cumRet.shape)
    drawdown = np.zeros(cumRet.shape)
    drawdownduration = np.zeros(cumRet.shape)

    for t in np.arange(1, cumRet.shape[0]):
        highwatermark[t] = np.maximum(highwatermark[t - 1], cumRet[t])
        drawdown[t] = (1 + cumRet[t]) / (1 + highwatermark[t]) - 1
        if drawdown[t] == 0:
            drawdownduration[t] = 0
        else:
            drawdownduration[t] = drawdownduration[t - 1] + 1

    maxDD, i = np.min(drawdown), int(np.argmin(drawdown))  # drawdown < 0 always
    maxDDD = float(np.max(drawdownduration))
    return float(maxDD), maxDDD, i
