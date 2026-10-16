"""
KO/PEP pairs trade with a Kalman-filter hedge ratio.

Design notes (each fixes a problem in an earlier version of this repo):
1. P&L comes from positions you could actually hold: n_y shares of KO against
   beta * n_y shares of PEP, with beta frozen at entry. The earlier version
   booked day-to-day changes in the filter's `spread`, which is its one-step
   prediction error, not a tradable price. Over a trade those changes add up to
   (error at exit) - (error at entry), so every completed trade showed a profit
   by construction, even on random data.
2. The trading z-score is scaled by the spread's own trailing 60-day volatility.
   With delta=1e-4 and Ve=1e-3 at KO/PEP price levels, the filter's variance
   estimate S is far too large, so z = e / sqrt(S) almost never reaches +/-2 and
   the strategy barely trades. (The filter's own z-score is kept as zscore_model.)
3. The stop-loss fires on the right side (a long spread is stopped out if z keeps
   falling).
4. position_size and transaction_cost_bps are applied: 5bp on both legs, on
   entry and on exit.
5. sanity_check() runs the same strategy on two unrelated random walks. There is
   nothing to find there, so its Sharpe ratios should centre on zero.

Run:  python kalman_pairs.py            (KO/PEP backtest + sanity check)
      python kalman_pairs.py --sanity   (sanity check only, no download)
"""
import sys
from dataclasses import dataclass
from typing import List

import numpy as np
import pandas as pd

# Fixed two-year window so the results are reproducible (END is exclusive)
START, END = "2024-10-13", "2026-10-13"


# ------------------------------------------------------------------ filter
class KalmanPairsFilter:
    """Time-varying regression y_t = alpha_t + beta_t * x_t + noise,
    with alpha and beta following random walks."""

    def __init__(self, delta: float = 1e-4, Ve: float = 1e-3):
        self.Ve = Ve
        self.A = np.eye(2)
        self.Q = delta * np.eye(2)
        self.theta = np.zeros(2)            # [alpha, beta]
        self.P = np.eye(2)

    def update(self, x: float, y: float):
        H = np.array([1.0, x])
        theta_pred = self.A @ self.theta
        P_pred = self.A @ self.P @ self.A.T + self.Q
        e = y - H @ theta_pred              # innovation: one-step prediction error
        S = H @ P_pred @ H + self.Ve        # innovation variance
        K = P_pred @ H / S
        self.theta = theta_pred + K * e
        I_KH = np.eye(2) - np.outer(K, H)
        self.P = I_KH @ P_pred @ I_KH.T + np.outer(K, K) * self.Ve   # Joseph form
        return e, S


def run_filter(prices: pd.DataFrame, y_col: str = "KO", x_col: str = "PEP",
               z_window: int = 60, **filter_params) -> pd.DataFrame:
    kf = KalmanPairsFilter(**filter_params)
    rows = []
    for date, y, x in zip(prices.index, prices[y_col].to_numpy(), prices[x_col].to_numpy()):
        e, S = kf.update(x, y)
        rows.append({"date": date, y_col: y, x_col: x,
                     "alpha": kf.theta[0], "beta": kf.theta[1],
                     "spread": e, "spread_var": S,
                     "absorbed": 1 - kf.Ve / S})   # share of today's error moved into alpha/beta
    df = pd.DataFrame(rows).set_index("date")
    df["zscore_model"] = df["spread"] / np.sqrt(df["spread_var"])     # the filter's own z-score
    # Trading signal: today's error over the volatility of the previous z_window errors
    df["zscore"] = df["spread"] / df["spread"].rolling(z_window).std().shift(1)
    return df


# ---------------------------------------------------------------- backtest
@dataclass
class BacktestConfig:
    entry_threshold: float = 2.0      # enter when |z| > this
    exit_threshold: float = 0.5       # exit once z is back inside +/- this
    stop_loss_z: float = 4.0          # stop out if z moves this far against you
    position_size: float = 10_000     # dollars in the KO leg; PEP leg is beta-weighted
    transaction_cost_bps: float = 5   # charged on both legs, on entry and on exit
    burn_in: int = 60                 # days for the filter and the z-score window to settle


@dataclass
class Trade:
    entry_date: pd.Timestamp
    exit_date: pd.Timestamp
    side: int               # +1 = long spread (long KO, short PEP), -1 = short spread
    entry_z: float
    exit_z: float
    pnl: float              # dollars, after costs
    holding_days: int


def backtest(df: pd.DataFrame, cfg: BacktestConfig = BacktestConfig(),
             y_col: str = "KO", x_col: str = "PEP") -> dict:
    df = df.iloc[cfg.burn_in:].copy()
    if df["zscore"].isna().any():
        raise ValueError("burn_in must be at least the z-score window")
    y, x = df[y_col].to_numpy(), df[x_col].to_numpy()
    z, beta = df["zscore"].to_numpy(), df["beta"].to_numpy()
    tc = cfg.transaction_cost_bps * 1e-4

    side, n_y, n_x = 0, 0.0, 0.0
    entry_i, entry_z, trade_pnl = 0, 0.0, 0.0
    daily = np.zeros(len(df))
    held = np.zeros(len(df), dtype=int)
    trades: List[Trade] = []

    for i in range(1, len(df)):
        zi = z[i - 1]                         # signal known at the close of day i-1
        cost = 0.0

        if side == 0:
            if abs(zi) > cfg.entry_threshold:
                side = 1 if zi < 0 else -1    # z < 0: KO cheap vs. PEP -> long spread
                n_y = cfg.position_size / y[i - 1]
                n_x = beta[i - 1] * n_y       # hedge ratio known at that close, then frozen
                cost = tc * (n_y * y[i - 1] + abs(n_x) * x[i - 1])
                entry_i, entry_z, trade_pnl = i, zi, -cost
        else:
            if side == 1:
                exit_now = zi > -cfg.exit_threshold or zi < -cfg.stop_loss_z
            else:
                exit_now = zi < cfg.exit_threshold or zi > cfg.stop_loss_z
            if exit_now:
                cost = tc * (n_y * y[i - 1] + abs(n_x) * x[i - 1])
                trades.append(Trade(df.index[entry_i - 1], df.index[i - 1], side,
                                    entry_z, zi, trade_pnl - cost, i - entry_i))
                side = 0

        # P&L from holding the position from close(i-1) to close(i)
        pnl = side * (n_y * (y[i] - y[i - 1]) - n_x * (x[i] - x[i - 1])) if side else 0.0
        trade_pnl += pnl
        daily[i] = pnl - cost
        held[i] = side

    out = df.iloc[1:].copy()
    out["daily_pnl"] = daily[1:]
    out["cum_pnl"] = out["daily_pnl"].cumsum()
    out["position"] = held[1:]

    pnl_series = out["daily_pnl"]
    sharpe = pnl_series.mean() / pnl_series.std() * np.sqrt(252) if pnl_series.std() > 0 else 0.0
    peak = out["cum_pnl"].cummax().clip(lower=0)      # drawdowns measured from zero P&L too
    max_dd = (peak - out["cum_pnl"]).max()

    return {
        "df": out,
        "trades": trades,
        "total_pnl": out["cum_pnl"].iloc[-1],
        "sharpe": sharpe,
        "max_drawdown": max_dd,
        "num_trades": len(trades),
        "win_rate": float(np.mean([t.pnl > 0 for t in trades])) if trades else 0.0,
        "avg_holding_days": float(np.mean([t.holding_days for t in trades])) if trades else 0.0,
        "still_open": side != 0,
    }


def print_results(res: dict, cfg: BacktestConfig) -> None:
    leg = cfg.position_size
    print("=== BACKTEST RESULTS (after costs) ===")
    print(f"Total P&L:     ${res['total_pnl']:,.0f}  ({res['total_pnl'] / leg:.1%} of the ${leg:,.0f} KO leg)")
    print(f"Sharpe ratio:  {res['sharpe']:.2f}")
    print(f"Max drawdown:  ${res['max_drawdown']:,.0f}  ({res['max_drawdown'] / leg:.1%} of the KO leg)")
    print(f"Trades:        {res['num_trades']}  (win rate {res['win_rate']:.0%}, "
          f"average hold {res['avg_holding_days']:.1f} days)")
    if res["still_open"]:
        print("Note: a position was still open at the end and isn't in the trade count.")
    print("\nFirst trades:")
    for t in res["trades"][:5]:
        print(f"  {t.entry_date.date()} -> {t.exit_date.date()}  {'LONG ' if t.side == 1 else 'SHORT'}  "
              f"z {t.entry_z:+.2f} -> {t.exit_z:+.2f}  P&L ${t.pnl:,.0f}")


# ------------------------------------------------------------ sanity check
def sanity_check(n_sims: int = 200, n_days: int = 504, seed: int = 0) -> np.ndarray:
    """Run the same filter and strategy on two unrelated random walks.
    There is no relationship to exploit, so Sharpe ratios should centre on zero
    (slightly below, after costs). A clearly positive median means a bug."""
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range("2020-01-01", periods=n_days)
    sharpes = []
    for _ in range(n_sims):
        prices = pd.DataFrame({
            "KO": 65 * np.exp(np.cumsum(rng.normal(0, 0.012, n_days))),
            "PEP": 160 * np.exp(np.cumsum(rng.normal(0, 0.013, n_days))),
        }, index=dates)
        sharpes.append(backtest(run_filter(prices))["sharpe"])
    return np.array(sharpes)


def main() -> None:
    if "--sanity" not in sys.argv:
        import yfinance as yf

        prices = yf.download(["KO", "PEP"], start=START, end=END,
                             auto_adjust=True, progress=False)["Close"].dropna()
        print(f"{len(prices)} trading days, {prices.index[0].date()} to {prices.index[-1].date()}")

        try:
            from statsmodels.tsa.stattools import coint
            p_value = coint(prices["KO"], prices["PEP"])[1]
            print(f"Engle-Granger cointegration p-value, KO on PEP: {p_value:.3f}")
        except ImportError:
            print("(pip install statsmodels to run the cointegration test)")

        cfg = BacktestConfig()
        df = run_filter(prices)
        d = df.iloc[cfg.burn_in:]
        print(f"\nShare of each day's error absorbed into alpha/beta: {d['absorbed'].median():.2%}")
        print(f"Std of the filter's own z-score (1.0 if well calibrated): {d['zscore_model'].std():.2f}")
        print(f"Days with |filter z| > 2: {(d['zscore_model'].abs() > 2).sum()} of {len(d)}\n")

        print_results(backtest(df, cfg), cfg)

    s = sanity_check()
    print(f"\nSanity check on {len(s)} pairs of unrelated random walks: median Sharpe "
          f"{np.median(s):.2f} (5th-95th percentile {np.percentile(s, 5):.2f} to {np.percentile(s, 95):.2f})")


if __name__ == "__main__":
    main()
