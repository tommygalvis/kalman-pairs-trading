# Kalman Filter Pairs Trading

Pairs trading on KO (Coca-Cola) and PEP (PepsiCo), using a Kalman filter to estimate a time-varying hedge ratio.

## Files

- `kalman_filter.py`: `KalmanPairsFilter` models `KO = alpha + beta * PEP + noise`, with alpha and beta following a random walk. It downloads 2 years of daily prices from Yahoo Finance, runs the filter, and builds a DataFrame with the hedge ratio, spread (innovation) and z-score.
- `backtest.py`: `PairsTradingBacktest` trades the z-score signal from `kalman_filter.py`.

## Trading rules

Defaults are in `BacktestConfig`:

| Rule | Condition |
|---|---|
| Enter long spread (long KO, short PEP) | z < −2.0 |
| Enter short spread (short KO, long PEP) | z > 2.0 |
| Exit | \|z\| < 0.5 |
| Stop-loss | z moves past ±4.0 against the position |

The first 50 days are skipped while the filter settles. Signals use the previous close, so there is no look-ahead in the entries.

## Usage

```bash
pip install -r requirements.txt
python backtest.py
```

The script prints the filter statistics, backtest metrics (P&L, Sharpe, max drawdown, win rate, holding period) and a sample of trades.

## Limitations

- P&L is measured in raw spread units. `position_size` and `transaction_cost_bps` are defined but not yet applied.
- The traded "spread" is the filter's one-step prediction error, not a tradable KO − β·PEP portfolio, so the results overstate what you could actually earn.
