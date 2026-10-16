# Kalman Filter Pairs Trading: KO / PEP

Tests a mean-reversion pairs trade on Coca-Cola (KO) and PepsiCo (PEP), using a Kalman filter to estimate a time-varying hedge ratio. **Result: no edge after costs**, consistent with a cointegration test that finds no long-run relationship between the two stocks.

## Results

Two years of daily data (14 Oct 2024 – 12 Oct 2026), 5 bp costs on both legs at entry and exit, $10,000 KO leg:

| Metric | Value |
|---|---|
| Engle–Granger cointegration p-value | 0.985 (no cointegration) |
| Sharpe ratio | −0.44 |
| Total P&L | −$437 (−4.4% of the KO leg) |
| Trades | 23 (win rate 52%, average hold 1.4 days) |
| Max drawdown | $1,216 |

**Sanity check:** the same strategy on 200 pairs of unrelated random walks gives a median Sharpe of −0.13, which is what an edge-free strategy should produce after costs. A clearly positive median there would point to a bug.

## How it works

- **Filter:** models `KO = alpha + beta × PEP + noise`, with alpha and beta following random walks.
- **Signal:** the filter's prediction error divided by its trailing 60-day volatility.
  - **Entry:** when |z| > 2, using the previous day's close, so there's no look-ahead.
  - **Exit:** when |z| < 0.5.
  - **Stop-loss:** when the move against the position reaches |z| = 4.
- **P&L:** marked on actual share positions. The trade holds n shares of KO against beta × n shares of PEP, with beta frozen at entry.

## Why there's no edge

- **No cointegration:** the test can't reject that KO and PEP drift apart, so there's no stable spread to mean-revert to.
- **The filter adapts too fast:** with these settings it absorbs 99.95% of each day's error into alpha and beta. The "spread" left over is mostly one-day noise, which is why trades last only 1.4 days on average. A slower filter (a smaller `delta`) would be the first thing to test.

## Earlier version

An earlier version of this repo measured P&L as day-to-day changes in the filter's prediction error rather than on actual positions. Every trade looked profitable by construction, even on random data. It also never applied costs, and its stop-loss could never trigger. The design notes at the top of `kalman_pairs.py` explain each fix.

## Usage

```bash
pip install -r requirements.txt
python kalman_pairs.py            # KO/PEP backtest + sanity check
python kalman_pairs.py --sanity   # sanity check only (no download)
```
