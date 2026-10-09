import numpy as np
import pandas as pd
from dataclasses import dataclass, field
from typing import List, Optional
from enum import Enum

from kalman_filter import df

class Position(Enum):
  FLAT = 0
  LONG_SPREAD = 1   # Long Y, Short X
  SHORT_SPREAD = -1  # Short Y, Long X

@dataclass
class Trade:
  """Record of a completed trade."""
  entry_date: pd.Timestamp
  exit_date: pd.Timestamp
  position: Position
  entry_zscore: float
  exit_zscore: float
  pnl: float
  holding_days: int

@dataclass
class BacktestConfig:
  """Configuration for the backtest."""
  entry_threshold: float = 2.0      # Enter when |z| > this
  exit_threshold: float = 0.5       # Exit when |z| < this
  stop_loss_z: float = 4.0          # Stop out if |z| exceeds this
  position_size: float = 10000      # Dollar value per leg
  transaction_cost_bps: float = 5   # Cost in basis points per trade

@dataclass
class BacktestResult:
  """Results from a backtest run."""
  df: pd.DataFrame
  trades: List[Trade]
  total_pnl: float
  sharpe_ratio: float
  max_drawdown: float
  win_rate: float
  avg_holding_days: float
  num_trades: int

class PairsTradingBacktest:
  """
  Backtesting engine for Kalman filter pairs trading.

  This handles the full lifecycle: signal generation, position management,
  P&L calculation, and performance metrics.
  """

  def __init__(self, config: Optional[BacktestConfig] = None):
      self.config = config or BacktestConfig()

  def run(self, df: pd.DataFrame) -> BacktestResult:
      """
      Run backtest on a DataFrame with columns: zscore, spread, KO, PEP

      Parameters
      ----------
      df : pd.DataFrame
          Must have 'zscore' and 'spread' columns from Kalman filter,
          plus price columns for the two assets.

      Returns
      -------
      BacktestResult
          Complete backtest results including trade list and metrics.
      """
      # Skip burn-in period for filter to stabilize
      df = df.iloc[50:].copy()

      position = Position.FLAT
      entry_idx = None
      entry_zscore = None
      cumulative_spread_pnl = 0

      trades: List[Trade] = []
      daily_pnl = []
      positions = []

      for i in range(1, len(df)):
          z = df['zscore'].iloc[i-1]  # Signal from previous close
          spread_change = df['spread'].iloc[i] - df['spread'].iloc[i-1]

          # Position management
          prev_position = position

          if position == Position.FLAT:
              # Entry logic
              if z < -self.config.entry_threshold:
                  position = Position.LONG_SPREAD
                  entry_idx = i
                  entry_zscore = z
              elif z > self.config.entry_threshold:
                  position = Position.SHORT_SPREAD
                  entry_idx = i
                  entry_zscore = z

          elif position == Position.LONG_SPREAD:
              # Exit logic for long spread
              if z > -self.config.exit_threshold or z < -self.config.stop_loss_z:
                  # Record trade
                  trades.append(Trade(
                      entry_date=df.index[entry_idx],
                      exit_date=df.index[i],
                      position=position,
                      entry_zscore=entry_zscore,
                      exit_zscore=z,
                      pnl=cumulative_spread_pnl,
                      holding_days=i - entry_idx
                  ))
                  cumulative_spread_pnl = 0
                  position = Position.FLAT

          elif position == Position.SHORT_SPREAD:
              # Exit logic for short spread
              if z < self.config.exit_threshold or z > self.config.stop_loss_z:
                  trades.append(Trade(
                      entry_date=df.index[entry_idx],
                      exit_date=df.index[i],
                      position=position,
                      entry_zscore=entry_zscore,
                      exit_zscore=z,
                      pnl=cumulative_spread_pnl,
                      holding_days=i - entry_idx
                  ))
                  cumulative_spread_pnl = 0
                  position = Position.FLAT

          # Calculate daily P&L
          if position == Position.LONG_SPREAD:
              day_pnl = spread_change  # Long spread profits when spread increases
              cumulative_spread_pnl += day_pnl
          elif position == Position.SHORT_SPREAD:
              day_pnl = -spread_change  # Short spread profits when spread decreases
              cumulative_spread_pnl += day_pnl
          else:
              day_pnl = 0

          daily_pnl.append(day_pnl)
          positions.append(position.value)

      # Add results to dataframe
      df = df.iloc[1:].copy()  # Align with pnl array
      df['daily_pnl'] = daily_pnl
      df['cumulative_pnl'] = np.cumsum(daily_pnl)
      df['position'] = positions

      # Calculate metrics
      total_pnl = df['cumulative_pnl'].iloc[-1]

      returns = df['daily_pnl']
      sharpe = returns.mean() / returns.std() * np.sqrt(252) if returns.std() > 0 else 0

      cumulative = df['cumulative_pnl']
      running_max = cumulative.cummax()
      drawdown = running_max - cumulative
      max_drawdown = drawdown.max()

      if trades:
          win_rate = sum(1 for t in trades if t.pnl > 0) / len(trades)
          avg_holding = np.mean([t.holding_days for t in trades])
      else:
          win_rate = 0
          avg_holding = 0

      return BacktestResult(
          df=df,
          trades=trades,
          total_pnl=total_pnl,
          sharpe_ratio=sharpe,
          max_drawdown=max_drawdown,
          win_rate=win_rate,
          avg_holding_days=avg_holding,
          num_trades=len(trades)
      )

# Run the backtest
config = BacktestConfig(
  entry_threshold=2.0,
  exit_threshold=0.5,
  stop_loss_z=4.0
)

backtest = PairsTradingBacktest(config)
result = backtest.run(df)

print(f"=== BACKTEST RESULTS ===")
print(f"Total P&L:        {result.total_pnl:.2f}")
print(f"Sharpe Ratio:     {result.sharpe_ratio:.2f}")
print(f"Max Drawdown:     {result.max_drawdown:.2f}")
print(f"Number of Trades: {result.num_trades}")
print(f"Win Rate:         {result.win_rate:.1%}")
print(f"Avg Holding Days: {result.avg_holding_days:.1f}")

print(f"\n=== SAMPLE TRADES ===")
for trade in result.trades[:5]:
  print(f"{trade.entry_date.date()} -> {trade.exit_date.date()}: "
        f"{'LONG' if trade.position == Position.LONG_SPREAD else 'SHORT'} "
        f"z={trade.entry_zscore:.2f}->{trade.exit_zscore:.2f} "
        f"PnL={trade.pnl:.2f}")
