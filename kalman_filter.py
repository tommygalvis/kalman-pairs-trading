import numpy as np
from dataclasses import dataclass
from typing import Tuple, Optional

@dataclass
class KalmanState:
      """Stores the current state of the Kalman filter."""
      theta: np.ndarray      # State estimate [alpha, beta]
      P: np.ndarray          # State covariance matrix
      spread: float          # Latest innovation (prediction error)
      spread_var: float      # Innovation variance
      zscore: float          # Standardized spread

class KalmanPairsFilter:
      """
      Kalman Filter for adaptive pairs trading.

      Models the relationship y_t = alpha_t + beta_t * x_t + noise
      where alpha and beta are time-varying parameters estimated
      by the filter.

      Parameters
      ----------
      delta : float
          State transition variance. Controls how quickly alpha and beta
          can change. Higher values = more responsive but noisier estimates.
          Typical range: 1e-5 to 1e-3

      Ve : float
          Observation noise variance. How much noise is in the price data.
          Higher values = smoother estimates, slower adaptation.
          Typical range: 1e-4 to 1e-2

      theta_init : np.ndarray, optional
          Initial state estimate [alpha, beta]. Default is [0, 0].

      P_init : np.ndarray, optional
          Initial state covariance. Default is identity matrix (high uncertainty).
      """

      def __init__(
          self,
          delta: float = 1e-4,
          Ve: float = 1e-3,
          theta_init: Optional[np.ndarray] = None,
          P_init: Optional[np.ndarray] = None
      ):
          self.delta = delta
          self.Ve = Ve

          # State dimension (alpha and beta)
          self.n_state = 2

          # State transition matrix (identity for random walk)
          self.A = np.eye(self.n_state)

          # State noise covariance
          self.Q = delta * np.eye(self.n_state)

          # Initialize state
          self.theta = theta_init if theta_init is not None else np.zeros(self.n_state)
          self.P = P_init if P_init is not None else np.eye(self.n_state)

          # Track history for analysis
          self.history = {
              'alpha': [],
              'beta': [],
              'spread': [],
              'spread_std': [],
              'zscore': []
          }

      def update(self, x: float, y: float) -> KalmanState:
          """
          Process a new observation and update state estimates.

          Parameters
          ----------
          x : float
              Price of the independent asset (e.g., PEP)
          y : float
              Price of the dependent asset (e.g., KO)

          Returns
          -------
          KalmanState
              Current state including spread and z-score for trading signals
          """
          # Observation matrix: y = [1, x] @ [alpha, beta]
          H = np.array([1.0, x])

          # === PREDICTION STEP ===
          # Predicted state (random walk: stays the same)
          theta_pred = self.A @ self.theta

          # Predicted covariance (increases due to state noise)
          P_pred = self.A @ self.P @ self.A.T + self.Q

          # === UPDATE STEP ===
          # Innovation (prediction error = spread)
          y_pred = H @ theta_pred
          e = y - y_pred

          # Innovation covariance
          S = H @ P_pred @ H.T + self.Ve

          # Kalman gain
          K = P_pred @ H.T / S

          # Updated state estimate
          self.theta = theta_pred + K * e

          # Updated covariance (Joseph form for numerical stability)
          I_KH = np.eye(self.n_state) - np.outer(K, H)
          self.P = I_KH @ P_pred @ I_KH.T + np.outer(K, K) * self.Ve

          # Compute trading signals
          spread = e
          spread_std = np.sqrt(S)
          zscore = spread / spread_std

          # Store history
          self.history['alpha'].append(self.theta[0])
          self.history['beta'].append(self.theta[1])
          self.history['spread'].append(spread)
          self.history['spread_std'].append(spread_std)
          self.history['zscore'].append(zscore)

          return KalmanState(
              theta=self.theta.copy(),
              P=self.P.copy(),
              spread=spread,
              spread_var=S,
              zscore=zscore
          )

      @property
      def alpha(self) -> float:
          """Current intercept estimate."""
          return self.theta[0]

      @property
      def beta(self) -> float:
          """Current hedge ratio estimate."""
          return self.theta[1]

      def get_history_df(self):
          """Return history as a pandas DataFrame."""
          import pandas as pd
          return pd.DataFrame(self.history)


import pandas as pd
import yfinance as yf
from datetime import datetime, timedelta

# Fetch two years of daily data for our pair
end_date = datetime.now()
start_date = end_date - timedelta(days=365*2)

data = yf.download(['KO', 'PEP'], start=start_date, end=end_date, progress=False)
prices = data['Close'].dropna()

print(f"Loaded {len(prices)} trading days")
print(f"Date range: {prices.index[0].date()} to {prices.index[-1].date()}")

# Initialize the Kalman filter
# delta=1e-4 means we expect alpha/beta to change slowly
# Ve=1e-3 reflects typical daily price noise
kf = KalmanPairsFilter(delta=1e-4, Ve=1e-3)

# Run the filter through all observations
results = []
for i in range(len(prices)):
  x = prices['PEP'].iloc[i]
  y = prices['KO'].iloc[i]
  state = kf.update(x, y)
  results.append({
      'date': prices.index[i],
      'KO': y,
      'PEP': x,
      'alpha': state.theta[0],
      'beta': state.theta[1],
      'spread': state.spread,
      'zscore': state.zscore
  })

df = pd.DataFrame(results).set_index('date')

# Show summary statistics
print(f"\nHedge ratio (beta):")
print(f"  Mean: {df['beta'].mean():.4f}")
print(f"  Std:  {df['beta'].std():.4f}")
print(f"  Min:  {df['beta'].min():.4f}")
print(f"  Max:  {df['beta'].max():.4f}")

print(f"\nZ-score statistics:")
print(f"  Mean: {df['zscore'].mean():.4f}")
print(f"  Std:  {df['zscore'].std():.4f}")
print(f"  Times |z| > 2: {(df['zscore'].abs() > 2).sum()}")
