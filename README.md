# Quantitative Portfolio Optimization & PCA Factor Backtesting

This repository contains a summer project with two connected parts:

1. **Portfolio construction**: Markowitz mean–variance optimization (efficient frontier / tangency portfolio) and a **Black–Litterman** posterior return model.
2. **Strategy backtesting**: A simple **PCA-based factor strategy** (daily re-fit) and a vectorized backtest with core performance metrics (Sharpe, cumulative return, max drawdown).

> Goal: demonstrate the full workflow **from theory → implementation → backtest → risk evaluation**, in an engineering-friendly structure.

---

## Repository structure

- `notebooks/`  
  Research notebooks (explanation + experiments).  
  - `01_markowitz_black_litterman.ipynb`  
  - `02_pca_factor_backtest.ipynb`

- `src/`  
  Reusable code (what you would keep in a real project)
  - `src/portfolio/markowitz.py` — efficient frontier, GMV, max-Sharpe
  - `src/portfolio/black_litterman.py` — implied returns & BL posterior
  - `src/factor/pca_strategy.py` — PCA factor signal → long/short positions
  - `src/backtest/engine.py` — backtest engine (positions → returns)
  - `src/backtest/performance.py` — Sharpe, cumulative return, max drawdown
  - `src/utils/io.py` — load wide price tables (Date + tickers)

- `data/`
  - `data/sample/` — small sample data for reproducible demos (recommended)
  - `data/raw/`, `data/processed/` — ignored by git (avoid pushing large files)

- `scripts/`
  - `run_portfolio.py` — CLI demo for Markowitz + BL
  - `run_backtest.py` — CLI demo for PCA strategy backtest

---

## Quickstart

### 1) Create environment & install dependencies

```bash
pip install -r requirements.txt
```

### 2) Run notebooks

Open and run:
- `notebooks/01_markowitz_black_litterman.ipynb`
- `notebooks/02_pca_factor_backtest.ipynb`

### 3) Run from command line (optional)

**PCA factor backtest**
```bash
python scripts/run_backtest.py --data data/sample/IJR_small.txt --lookback 252 --num-factors 5 --top-n 50
```

**Markowitz + Black–Litterman demo**
```bash
python scripts/run_portfolio.py --data data/sample/prices.csv
```

---

## Method overview

### Markowitz (mean–variance)

Given expected returns **μ** and covariance **Σ**, solve:
- **GMV**: minimize \( w^\top \Sigma w \) s.t. \( \sum w = 1 \)
- **Efficient frontier**: minimize variance s.t. \( w^\top \mu = r_{target},\ \sum w = 1 \)
- **Tangency**: maximize Sharpe ratio \( (w^\top \mu - r_f) / \sqrt{w^\top \Sigma w} \)

### Black–Litterman

- Prior (implied) equilibrium excess returns:
  \[
  \pi = \delta \Sigma w_{mkt}
  \]
- Add views \( (P, q, \Omega) \) to obtain posterior mean:
  \[
  \mu_{BL} = \left[(\tau\Sigma)^{-1} + P^\top\Omega^{-1}P\right]^{-1}
            \left[(\tau\Sigma)^{-1}\pi + P^\top\Omega^{-1}q\right]
  \]

### PCA strategy (backtest)

Daily re-fit using a rolling window:
1) compute daily returns  
2) run PCA on the return panel to obtain factors  
3) multi-output regression of asset returns on factors  
4) rank assets by cumulative predicted return in the window  
5) long top-N and short bottom-N assets

Performance metrics include cumulative compounded return and **maximum drawdown**.

---

## Notes on data

The original factor notebook uses an ETF constituent panel (wide table: `Date` + many tickers).  
To keep the repo lightweight, large datasets should **not** be committed. Use:
- `data/sample/` for small demo files
- a short note in the notebook/README describing how to obtain full data

---

## License

This project is licensed under the MIT License.
