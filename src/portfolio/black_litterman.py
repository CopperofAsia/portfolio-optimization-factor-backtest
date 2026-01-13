"""
Black–Litterman model utilities.

Conventions:
- cov: covariance matrix of asset returns (Σ)
- w_mkt: market-cap weights (sums to 1)
- delta: risk aversion coefficient (δ)
- tau: scaling parameter for the prior covariance uncertainty (τ)
- P, q, Omega: view matrix, view returns, and view uncertainty

References:
- He & Litterman (1999), "The Intuition Behind Black-Litterman Model Portfolios"
"""

from __future__ import annotations

from typing import Tuple, Optional

import numpy as np


def implied_equilibrium_returns_pi(delta: float, cov: np.ndarray, w_mkt: np.ndarray) -> np.ndarray:
    """
    Compute implied equilibrium excess returns (π = δ Σ w_mkt).
    """
    return float(delta) * (cov @ w_mkt)


def estimate_delta_from_market(mu_mkt: float, var_mkt: float, rf: float = 0.0) -> float:
    """
    Estimate risk aversion coefficient using:
      δ = (E[R_m] - rf) / Var(R_m)

    Parameters
    ----------
    mu_mkt : expected market return (not excess unless rf=0)
    var_mkt: market return variance
    rf     : risk-free rate
    """
    if var_mkt <= 0:
        raise ValueError("var_mkt must be positive.")
    return float((mu_mkt - rf) / var_mkt)


def black_litterman_posterior(
    cov: np.ndarray,
    pi: np.ndarray,
    P: np.ndarray,
    q: np.ndarray,
    Omega: np.ndarray,
    tau: float = 0.05,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute the Black–Litterman posterior mean and covariance.

    Posterior:
      μ_BL = [ (τΣ)^(-1) + P^T Ω^(-1) P ]^(-1) [ (τΣ)^(-1) π + P^T Ω^(-1) q ]
      Σ_BL = Σ + [ (τΣ)^(-1) + P^T Ω^(-1) P ]^(-1)

    Returns
    -------
    mu_bl  : posterior mean (same dimension as pi)
    cov_bl : posterior covariance
    """
    cov = np.asarray(cov)
    pi = np.asarray(pi).reshape(-1, 1)
    P = np.asarray(P)
    q = np.asarray(q).reshape(-1, 1)
    Omega = np.asarray(Omega)

    if tau <= 0:
        raise ValueError("tau must be positive.")
    if cov.shape[0] != cov.shape[1]:
        raise ValueError("cov must be square.")
    n = cov.shape[0]
    if pi.shape[0] != n:
        raise ValueError("pi dimension mismatch.")
    if P.shape[1] != n:
        raise ValueError("P must have n columns.")
    if Omega.shape[0] != Omega.shape[1] or Omega.shape[0] != P.shape[0]:
        raise ValueError("Omega must be square and match number of views (rows of P).")

    tau_cov = tau * cov
    inv_tau_cov = np.linalg.inv(tau_cov)
    inv_Omega = np.linalg.inv(Omega)

    middle = inv_tau_cov + P.T @ inv_Omega @ P
    middle_inv = np.linalg.inv(middle)

    mu_bl = middle_inv @ (inv_tau_cov @ pi + P.T @ inv_Omega @ q)
    cov_bl = cov + middle_inv

    return mu_bl.ravel(), cov_bl
