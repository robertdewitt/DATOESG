import gymnasium as gym
from gymnasium import spaces
import numpy as np
import torch
import logging 
from typing import Optional, Tuple, Dict, Any
import platform
import pandas as pd

# Set up logger for this module
logger = logging.getLogger(__name__)

class MultiOrderExecutionEnv(gym.Env):
    metadata = {"render_modes": ["human"]}

    def __init__(self, stock_df_list, orders_df, impact_coef, decay_rate, num_envs, min_rate=0.0, max_rate=0.1, window_size=1, unfilled_penalty=1e6, device: Optional[str] = None, render_mode=None):
        """
        Initialize the Vectorized Optimal Execution Environment for reinforcement learning.
        @param stock_df_list: List of DataFrames containing stock data (e.g., VWAP, volume).
        @param orders_df: DataFrame containing orders (e.g., order_qty, time_horizon, side, etc.).
        @param impact_coef: Immediate impact coefficient (γ).
        @param decay_rate: Residual decay factor (κ).
        @param num_envs: Number of parallel environments (orders) to simulate.
        @param min_rate: Minimum fraction of volume that can be traded in one step.
        @param max_rate: Maximum fraction of volume that can be traded in one step.
        @param window_size: Number of steps to consider for residual impact decay.
        @param unfilled_penalty: Penalty applied to reward if order not fully executed by horizon.
        @param device: Device to use for tensor operations ('cuda', 'mps', or 'cpu').
        @param render_mode: Render mode for the environment.
        """
        super().__init__()
        self.render_mode = render_mode
        self.num_envs = num_envs

        # Device selection: explicit > MPS > CUDA > CPU
        if device:
            self.device = device
        elif torch.backends.mps.is_available():
            self.device = 'mps'
        elif torch.cuda.is_available():
            self.device = 'cuda'
        else:
            self.device = 'cpu'
        logger.info(f"Using device: {self.device}")

        # 1) Save parameters
        self.stock_df_list = stock_df_list
        self.orders_df = orders_df
        self.impact_coef = torch.tensor(impact_coef, device=self.device, dtype=torch.float32)
        self.decay_rate = torch.tensor(decay_rate, device=self.device, dtype=torch.float32)
        self.window_size = window_size
        self.min_rate = min_rate
        self.max_rate = max_rate
        self.unfilled_penalty = unfilled_penalty
        
        # Vectorized state variables
        self.immediate_impact = torch.zeros(self.num_envs, device=self.device, dtype=torch.float32)
        self.order_vwap = torch.zeros(self.num_envs, device=self.device, dtype=torch.float32)
        self.last_fill_price = torch.zeros(self.num_envs, device=self.device, dtype=torch.float32)
        self.last_trade_size = torch.zeros(self.num_envs, device=self.device, dtype=torch.int64)
        self.last_action_fraction = torch.zeros(self.num_envs, device=self.device, dtype=torch.float32)
        # Track the last step at which a non-zero trade occurred for each environment.
        # Initialise to -1 so that the first trade uses Δt = 1.
        self.last_trade_step = torch.full((self.num_envs,), -1, device=self.device, dtype=torch.int32)
   
        # 2) Define discrete action space with specified trade fractions, moved to device
        self.action_values = torch.tensor([0.0, 0.05, 0.1, 0.15, 0.25, 0.5, 0.75], dtype=torch.float32, device=self.device)
        self.action_space = spaces.Discrete(len(self.action_values))

        # 3) Define observation space for a single environment. reset() will return a batch of observations.
        obs_dim = 14  # Updated to match actual observation vector size
        # Define reasonable bounds for each observation component
        # [mid_price, volume, time_remaining, shares_remaining, order_adv_pct, signal,
        #  last_fill_price, last_trade_size, immediate_impact, aggregated_impact, arrival_price, regime, lag1 vol, 5 day avg 
        obs_low = np.array([
            0.0,      # mid_price (prices are positive)
            0.0,      # volume (volumes are positive)
            0.0,      # time_remaining (time is positive)
            0.0,      # shares_remaining (shares are positive)
            0.0,      # order_adv_pct (percentage is positive)
            -1.0,     # signal (normalized between -1 and 1)
            0.0,      # last_fill_price (prices are positive)
            0.0,      # last_trade_size (sizes are positive)
            0.0,      # immediate_impact (can be negative for sells)
            0.0,      # aggregated_impact (can be negative for sells)
            0.0,      # arrival_price (prices are positive)
            -1.0,     # regime (normalized between -1 and 1)
            0.0,      # lag1 vol (daily volatility lag1, can be zero)
            0.0       # 5 day avg (daily volatility 5 day, can be zero)
        ], dtype=np.float32)
        
        obs_high = np.array([
            1e10,     # mid_price (reasonable max price)
            1e16,     # volume (reasonable max volume)
            600.0,    # time_remaining (max trading minutes in a day)
            1e16,     # shares_remaining (reasonable max shares)
            1.0,      # order_adv_pct (max 100%)
            1.0,      # signal (normalized between -1 and 1)
            1e10,     # last_fill_price (reasonable max price)
            1e16,     # last_trade_size (reasonable max size)
            1e10,     # immediate_impact (reasonable max impact)
            1e10,     # aggregated_impact (reasonable max impact)
            1e10,     # arrival_price (reasonable max price)
            1.0,      # regime (normalized between -1 and 1)
            1e10,     # lag1 vol (daily volatility lag1, can be large)
            1e10      # 5 day avg (daily volatility 5 day, can be large)
        ], dtype=np.float32)
        
        self.observation_space = spaces.Box(low=obs_low, high=obs_high, shape=(obs_dim,), dtype=np.float32)

        # 4) Initialize state (will be tensors of size num_envs)
        self.current_step = torch.zeros(self.num_envs, device=self.device, dtype=torch.int32)
        self.shares_remaining = torch.zeros(self.num_envs, device=self.device, dtype=torch.int64)
        self.arrival_price = torch.zeros(self.num_envs, device=self.device, dtype=torch.float32)
        self.accumulated_impact = torch.zeros(self.num_envs, device=self.device, dtype=torch.float32)
        self.done = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)

        # These will also be tensors, initialized in reset
        self.order_idx = torch.zeros(self.num_envs, device=self.device, dtype=torch.int64)
        self.tickers = [None] * self.num_envs
        self.order_qty = torch.zeros(self.num_envs, device=self.device, dtype=torch.int64)
        self.adv_pct = torch.zeros(self.num_envs, device=self.device, dtype=torch.float32)
        self.adv = torch.zeros(self.num_envs, device=self.device, dtype=torch.float32)
        self.start_time = torch.zeros(self.num_envs, device=self.device, dtype=torch.int64)
        self.end_time = torch.zeros(self.num_envs, device=self.device, dtype=torch.int64)
        self.time_horizon = torch.zeros(self.num_envs, device=self.device, dtype=torch.int64)
        self.side = torch.zeros(self.num_envs, device=self.device, dtype=torch.int8)

        # We need to handle data for multiple tickers efficiently
        self.data_arrays = {}
        self._preload_data_arrays()

    def _preload_data_arrays(self):
        """
        Convert all stock dataframes into a dictionary of tensors for faster access.
        """
        for ticker, df_raw in self.stock_df_list.items():
            if isinstance(df_raw.columns, pd.MultiIndex):
                df = df_raw.xs(ticker, axis=1, level=1)
            else:
                df = df_raw
            
            self.data_arrays[ticker] = {
                'High': torch.tensor(df['High'].to_numpy(dtype=np.float32), device=self.device),
                'Low': torch.tensor(df['Low'].to_numpy(dtype=np.float32), device=self.device),
                'Open': torch.tensor(df['Open'].to_numpy(dtype=np.float32), device=self.device),
                'Close': torch.tensor(df['Close'].to_numpy(dtype=np.float32), device=self.device),
                'Volume': torch.tensor(df['Volume'].to_numpy(dtype=np.float32), device=self.device),
                'VWAP': torch.tensor(df['VWAP'].to_numpy(dtype=np.float32) if 'VWAP' in df.columns else ((df['High'] + df['Low'] + df['Close'] + df['Open']) / 4.0).to_numpy(dtype=np.float32), device=self.device),
                'Signal': torch.tensor(df['Signal'].to_numpy(dtype=np.float32) if 'Signal' in df.columns else np.zeros(len(df), dtype=np.float32), device=self.device),
                'Regime': torch.tensor(df['Regime'].to_numpy(dtype=np.float32) if 'Regime' in df.columns else np.zeros(len(df), dtype=np.float32), device=self.device),
                'DailyVol': torch.tensor(df['DailyVol'].to_numpy(dtype=np.float32) if 'DailyVol' in df.columns else np.zeros(len(df), dtype=np.float32), device=self.device),
                'DailyVolLag1': torch.tensor(df['DailyVolLag1'].to_numpy(dtype=np.float32) if 'DailyVolLag1' in df.columns else np.zeros(len(df), dtype=np.float32), device=self.device),
                'DailyVol5d': torch.tensor(df['DailyVol5d'].to_numpy(dtype=np.float32) if 'DailyVol5d' in df.columns else np.zeros(len(df), dtype=np.float32), device=self.device)
            }

    def reset(self, seed=None, options=None):
        """
        Begin a new episode by selecting a random batch of orders from orders_df.
        @param seed: Random seed for reproducibility.
        @param options: Additional options (not used here, but can be extended).
        @return: Initial observation tensor and a list of info dictionaries.
        """
        super().reset(seed=seed)
        
        # Set up both numpy and torch random number generators
        if seed is not None:
            np.random.seed(seed)
            torch.manual_seed(seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed(seed)
                torch.cuda.manual_seed_all(seed)
                torch.backends.cudnn.deterministic = True
                torch.backends.cudnn.benchmark = False

        # 1) Pick a random batch of order indices
        num_orders = len(self.orders_df)
        # Always sample random order indices for consistency
        order_indices = np.random.randint(0, num_orders, size=self.num_envs)
        
        self.order_idx = torch.tensor(order_indices, device=self.device)
        order_rows = self.orders_df.iloc[order_indices]

        # Extract order details for the batch
        self.tickers = order_rows['ticker'].tolist()
        self.order_qty = torch.tensor(order_rows['order_qty'].values, device=self.device, dtype=torch.int64)
        self.adv_pct = torch.tensor(order_rows['adv_pct'].values, device=self.device, dtype=torch.float32)
        # Extract adv values robustly (handling sequences or pandas Series)
        adv_list = []
        for x in order_rows['adv']:
            if isinstance(x, pd.Series):
                # Use iloc for positional indexing
                adv_list.append(float(x.iloc[0]))
            elif isinstance(x, (list, tuple, np.ndarray)):
                adv_list.append(float(x[0]))
            else:
                adv_list.append(float(x))
        self.adv = torch.tensor(adv_list, device=self.device, dtype=torch.float32)
        self.start_time = torch.tensor(order_rows['start_time'].values, device=self.device, dtype=torch.int64)
        self.end_time = torch.tensor(order_rows['end_time'].values, device=self.device, dtype=torch.int64)
        self.time_horizon = torch.tensor(order_rows['time_horizon'].values, device=self.device, dtype=torch.int64)
        self.side = torch.tensor([1 if s.lower() == 'buy' else -1 for s in order_rows['side']], device=self.device, dtype=torch.int8)
        
        if 'order_vwap' in order_rows.columns:
            order_vwap_values = order_rows['order_vwap'].fillna(0.0).values
        else:
            order_vwap_values = np.zeros(self.num_envs, dtype=np.float32)
        self.order_vwap = torch.tensor(order_vwap_values, device=self.device, dtype=torch.float32)

        # Initialize environment‐state variables
        self.current_step = torch.zeros(self.num_envs, device=self.device, dtype=torch.int32)
        self.shares_remaining = self.order_qty.clone()
        self.accumulated_impact = torch.zeros(self.num_envs, device=self.device, dtype=torch.float32)
        self.done = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self.last_fill_price = torch.zeros(self.num_envs, device=self.device, dtype=torch.float32)
        self.last_trade_size = torch.zeros(self.num_envs, device=self.device, dtype=torch.int64)
        self.last_action_fraction = torch.zeros(self.num_envs, device=self.device, dtype=torch.float32)
        self.immediate_impact = torch.zeros(self.num_envs, device=self.device, dtype=torch.float32)

        # Reset last trade step tracker
        self.last_trade_step = torch.full((self.num_envs,), -1, device=self.device, dtype=torch.int32)

        # Compute arrival_price for each order in the batch (vectorized)
        self.arrival_price = torch.zeros(self.num_envs, device=self.device, dtype=torch.float32)
        for ticker in set(self.tickers):
            env_indices = [i for i, t in enumerate(self.tickers) if t == ticker]
            env_indices_tensor = torch.tensor(env_indices, device=self.device, dtype=torch.long)
            
            start_indices = self.start_time[env_indices_tensor]

            # Data validation
            max_len = len(self.data_arrays[ticker]['High'])
            if (start_indices < 0).any() or (start_indices + self.time_horizon[env_indices_tensor] > max_len).any():
                 # Find the problematic order for a better error message
                for i in env_indices:
                    if self.start_time[i] < 0 or self.start_time[i] + self.time_horizon[i] > max_len:
                        raise ValueError(
                            f"Order {self.order_idx[i]} (ticker={ticker}) has start_time {self.start_time[i]} "
                            f"and time_horizon {self.time_horizon[i]}, which exceed data length {max_len}."
                        )

            highs = self.data_arrays[ticker]['High'][start_indices]
            lows = self.data_arrays[ticker]['Low'][start_indices]
            self.arrival_price[env_indices_tensor] = (highs + lows) / 2.0

        # Initialize first step values to avoid NaNs
        market_data = self._get_market_data_batch(['Volume', 'VWAP', 'High', 'Low', 'DailyVolLag1'], use_prior_step=False)
        current_volumes = market_data['Volume']
        vwap_prices = market_data['VWAP']
        mid_prices = (market_data['High'] + market_data['Low']) * 0.5
        daily_sigma = market_data['DailyVolLag1']
        
        # Set initial values for first step
        self.last_fill_price = vwap_prices.clone()
        self.last_trade_size = torch.zeros_like(current_volumes, dtype=torch.int64)
        self.immediate_impact = torch.zeros_like(current_volumes)
        self.accumulated_impact = torch.zeros_like(current_volumes)
        self.order_vwap = vwap_prices.clone()

        # Build and return the initial observation
        obs = self._get_observation()
        
        # Build the initial info dict with fields matching step() outputs
        infos = [{} for _ in range(self.num_envs)]
        for i in range(self.num_envs):
            infos[i] = {
                'order_idx': self.order_idx[i].item(),
                'ticker': self.tickers[i],
                'order_qty': self.order_qty[i].item(),
                'adv_pct': self.adv_pct[i].item(),
                'start_time': self.start_time[i].item(),
                'end_time': self.end_time[i].item(),
                'time_horizon': self.time_horizon[i].item(),
                'side': 'buy' if self.side[i].item() == 1 else 'sell',
                'arrival_price': self.arrival_price[i].item(),
                'order_vwap': self.order_vwap[i].item(),
                'shares_remaining': self.shares_remaining[i].item(),
                'current_step': self.current_step[i].item(),
                'immediate_impact': self.immediate_impact[i].item(),
                'accumulated_impact': self.accumulated_impact[i].item(),
                'last_fill_price': self.last_fill_price[i].item(),
                'last_trade_size': self.last_trade_size[i].item(),
                'vwap_price': vwap_prices[i].item(),
                'mid_price': mid_prices[i].item(),
                'total_reward': 0.0,
                'current_volume': current_volumes[i].item(),
                'action_percentage': 0.0,  # Initialize to 0 for first step
                'adv': self.adv[i].item()
            }
        
        # Always return a single observation when used with PPO
        return obs[0], infos[0]
   
    def _get_market_data_batch(self, fields: list, use_prior_step: bool = False) -> Dict[str, torch.Tensor]:
        """
        Gathers a batch of market data for the current step for specified fields.
        @param fields: List of market data fields to fetch
        @param use_prior_step: If True, returns data from previous step (for action selection)
        """
        indices = self.start_time + self.current_step
        if use_prior_step:
            indices = indices - 1  # Use previous step's data for action selection
        
        # Initialize result tensors
        results = {field: torch.zeros(self.num_envs, device=self.device, dtype=torch.float32) for field in fields}
        
        # Group environments by ticker
        for ticker in set(self.tickers):
            # Find which environments (indices) in the batch use this ticker
            env_indices = [i for i, t in enumerate(self.tickers) if t == ticker]
            env_indices_tensor = torch.tensor(env_indices, device=self.device, dtype=torch.long)

            # Get the corresponding indices into the market data arrays for this ticker
            market_indices = indices[env_indices_tensor]

            # Gather data for all requested fields for this ticker
            for field in fields:
                if field in self.data_arrays[ticker]:
                    results[field][env_indices_tensor] = self.data_arrays[ticker][field][market_indices]

        return results

    def _get_observation(self):
        """
        Return a batch of observations using data from the prior step for action selection.
        """
        # Gather data from prior step for action selection
        market_data = self._get_market_data_batch(['High', 'Low', 'Volume', 'Signal', 'Regime', 'DailyVolLag1', 'DailyVol5d'], use_prior_step=True)
        
        # Perform all operations on GPU
        mid_price = (market_data['High'] + market_data['Low']) * 0.5
        volume = market_data['Volume']
        signal = market_data['Signal']
        regime = market_data['Regime']
        vol_lag1 = market_data['DailyVolLag1']
        vol5d = market_data['DailyVol5d']

        # Create observation tensor from the gathered lists
        obs_tensor = torch.stack([
            mid_price,
            volume,
            (self.time_horizon - self.current_step).float(),
            self.shares_remaining,
            self.adv_pct,
            signal,
            self.last_fill_price,
            self.last_trade_size,
            self.immediate_impact,
            self.accumulated_impact,
            self.arrival_price,
            regime,
            vol_lag1,
            vol5d
        ], dim=1)


        # Only convert to numpy at the very end
        obs_numpy = obs_tensor.cpu().numpy()
            
        # Return as numpy array for gymnasium compatibility
        return obs_numpy


    def step(self, action):
        """
        Execute one step in the environment for a batch of orders.
        Uses current step data for execution and results.
        Continues until the end of the horizon even if the order is completed early.
        @param action: Action to take, either an integer or a list of fractions.
        @return: Tuple of (observation, reward, done, truncated, info)
        """
        if self.done.all():
            raise RuntimeError("All episodes are already done")

        # Initialize total_cost if not exists
        if not hasattr(self, 'total_cost'):
            self.total_cost = torch.zeros(self.num_envs, device=self.device, dtype=torch.float32)

        # Convert action to tensor on GPU
        if isinstance(action, (int, np.integer)):
            action = torch.tensor([action], device=self.device, dtype=torch.int64)
        else:
            action = torch.tensor(action, device=self.device, dtype=torch.int64)

        # Ensure action is valid
        if not (0 <= action < len(self.action_values)).all():
            raise ValueError(f"Invalid action {action}. Must be between 0 and {len(self.action_values)-1}")

        # 1) Get current market data for execution
        market_data = self._get_market_data_batch(['Volume', 'VWAP', 'High', 'Low', 'DailyVolLag1'], use_prior_step=False)
        current_volumes = market_data['Volume']
        vwap_prices = market_data['VWAP']
        mid_prices = (market_data['High'] + market_data['Low']) * 0.5
        daily_sigma = market_data['DailyVolLag1']

        # 2) Compute trade sizes for the batch using current volume
        fractions = self.action_values[action]
        # Only execute trades if there are shares remaining
        trade_sizes = torch.where(self.shares_remaining > 0, 
                                torch.min(fractions * current_volumes, self.shares_remaining),
                                torch.zeros_like(self.shares_remaining))
        trade_sizes = torch.round(trade_sizes).to(torch.int64)  # Round to nearest integer



        #  Compute impacts and fill prices using current market data
        self.immediate_impact = self.impact_coef * daily_sigma * torch.sqrt(trade_sizes.float() / self.adv)

        # --- Residual impact decay with variable Δt ---------------------------------
        # Time (in steps) since the last non-zero trade for every environment.
        delta_t = torch.where(
            self.last_trade_step >= 0,
            (self.current_step - self.last_trade_step).float(),
            torch.ones_like(self.current_step, dtype=torch.float32)
        )

        # Convert the user supplied decay_rate into a time constant τ if it is already
        # a per-step decay factor. This implementation interprets `decay_rate` as τ
        # directly, i.e. λ(Δt) = exp(-Δt / τ).
        lambda_decay = torch.exp(-delta_t / self.decay_rate)

        # Update the accumulated impact with the decayed residual plus immediate impact.
        self.accumulated_impact = lambda_decay * self.accumulated_impact + self.immediate_impact
        fill_prices = vwap_prices * (1 + self.side * self.accumulated_impact)

        # 4) Update order VWAP for the batch
        filled_qty = self.order_qty - self.shares_remaining
        total_qty = filled_qty + trade_sizes
        
        # Avoid division by zero for environments with zero total quantity
        safe_total_qty = torch.where(total_qty > 0, total_qty.float(), torch.ones_like(total_qty, dtype=torch.float32))
        
        # Update order VWAP only if we have shares remaining
        new_order_vwap = (self.order_vwap * filled_qty.float() + fill_prices * trade_sizes.float()) / safe_total_qty
        self.order_vwap = torch.where(self.shares_remaining > 0, new_order_vwap, self.order_vwap)

        # 5) Update state for the batch
        prev_shares = self.shares_remaining.clone()
        self.shares_remaining = torch.max(prev_shares - trade_sizes, torch.zeros_like(prev_shares))

        # Update last_trade_step for envs where we executed a non-zero quantity *before*
        # incrementing the global step counter so that Δt is computed correctly in the
        # next call to `step`.
        trade_executed_mask = trade_sizes > 0
        self.last_trade_step[trade_executed_mask] = self.current_step[trade_executed_mask]

        # Advance the environment time step and record additional state variables.
        self.current_step += 1
        self.last_fill_price = fill_prices
        self.last_trade_size = trade_sizes
        self.last_action_fraction = fractions

        # 6) Compute rewards for the batch
        reward = torch.zeros(self.num_envs, device=self.device)
        
        # Calculate slippage penalty - convert to basis points
        slippage = -self.side * (fill_prices - self.arrival_price) / self.arrival_price 
        
        # Calculate trade cost (slippage * trade size)
        trade_cost = slippage * trade_sizes.float()

        # Calculate EHV-based penalty (always negative)
        # EHV rate = (order % of ADV * 390) / horizon
        ehv_rate = (self.adv_pct * 390.0) / self.time_horizon.float()
        
        # Calculate deviation from EHV rate
        # For each 10% deviation, add 100 penalty (always negative)
        deviation = torch.abs(fractions - ehv_rate)
        ehv_penalty = -(deviation / 0.1) * 100.0  # Negative penalty per 10% deviation
        
        # During execution, combine trade cost and EHV penalty
        reward = trade_cost + ehv_penalty

        # Update total cost (accumulate raw costs)
        self.total_cost = self.total_cost + reward

        # Check for finished orders (terminated)
        terminated_mask = self.shares_remaining <= 0
        if terminated_mask.any():
            # For finished orders, keep the accumulated raw costs
            pass
        
        # Check for time horizon reached (truncated)
        # Only truncate if we've exceeded the time horizon
        truncated_mask = self.current_step >= self.time_horizon
        
        # Set done flag only for truncated orders
        self.done = truncated_mask

        # 7) Get new observation for the batch
        obs = self._get_observation()
        
        # 8) Prepare info dicts
        infos = [{} for _ in range(self.num_envs)]
        for i in range(self.num_envs):
            # Use 0.0 for order_vwap if it is NaN or missing
            order_vwap_val = self.order_vwap[i].item()
            mid_price = mid_prices[i].item()
            current_volume = current_volumes[i].item()
            total_reward = total_cost[i].item()
            vwap_price = vwap_prices[i].item()
            if np.isnan(order_vwap_val):
                order_vwap_val = 0.0
            if np.isnan(mid_price):
                mid_price = 0.0
            if np.isnan(current_volume):
                current_volume = 0.0
            if np.isnan(total_reward):
                total_reward = 0.0
            if np.isnan(vwap_price):
                vwap_price = 0.0
            
        
            # Always include market data
            market_info = {
                'vwap_price': vwap_price,
                'mid_price': mid_price,
                'current_volume': current_volume,
                'current_step': self.current_step[i].item(),
                'adv': self.adv[i].item(),
                'total_reward': total_reward  # Always include total_reward in market info
            }
            
            # Include order info only if the order is not completed
            if not terminated_mask[i]:
                order_info = {
                    'order_idx': self.order_idx[i].item(),
                    'ticker': self.tickers[i],
                    'order_qty': self.order_qty[i].item(),
                    'adv_pct': self.adv_pct[i].item(),
                    'adv': self.adv[i].item(),
                    'start_time': self.start_time[i].item(),
                    'end_time': self.end_time[i].item(),
                    'time_horizon': self.time_horizon[i].item(),
                    'side': 'buy' if self.side[i].item() == 1 else 'sell',
                    'shares_remaining': self.shares_remaining[i].item(),
                    'last_fill_price': self.last_fill_price[i].item(),
                    'last_trade_size': self.last_trade_size[i].item(),
                    'immediate_impact': self.immediate_impact[i].item(),
                    'accumulated_impact': self.accumulated_impact[i].item(),
                    'arrival_price': self.arrival_price[i].item(),
                    'order_vwap': order_vwap_val,
                    'total_reward': total_reward,
                    'action_percentage': self.last_action_fraction.item(),
                    'is_finished': terminated_mask[i].item()
                }
                infos[i] = {**market_info, **order_info}
            else:
                infos[i] = {**market_info,
                               'time_horizon': self.time_horizon[i].item(),
                               'ticker': self.tickers[i],
                               'side': 'buy' if self.side[i].item() == 1 else 'sell',
                               'shares_remaining': self.shares_remaining[i].item(),
                               'last_trade_size': self.last_trade_size[i].item(),
                               'last_fill_price': self.last_fill_price[i].item(),
                               'order_vwap': order_vwap_val,
                               'accumulated_impact': self.accumulated_impact[i].item(),
                               'is_finished': True}
        
        # Convert reward tensor to numpy for scalar extraction
        rewards_np = reward.cpu().numpy()

        # Extract Python-native scalars for return values
        obs_return = obs[0]
        reward_return = float(rewards_np[0])
        terminated_return = bool(terminated_mask[0].item())
        truncated_return = bool(truncated_mask[0].item())
        info_return = infos[0]

        # Return a single observation, scalar reward, Python bools for terminated/truncated, and info
        # Save last_info for rendering
        self.last_info = info_return
        return obs_return, reward_return, terminated_return, truncated_return, info_return

    def render(self):
        """
        Render the last step's info for debugging.
        """
        if not hasattr(self, 'last_info'):
            return
        info = self.last_info
        logger.debug(
            f"Step {info['current_step']}/{info['time_horizon']} | "
            f"Ticker: {info['ticker']} | MidPrice: {info['mid_price']:.2f} | "
            f"Shares Rem: {info['shares_remaining']:.2f} | "
            f"Last Trade Size: {info['last_trade_size']:.2f} | "
            f"Last Fill Price: {info['last_fill_price']:.2f} | "
            f"Side: {info['side'].capitalize()}"
        )

    
    def execute_orders(self, model, num_episodes=10):
        """
        Execute a batch of orders using the provided model.
        @param model: The trained RL model to use for action selection.
        @param num_orders: Number of orders to execute in this run.
        @return: List of order information dictionaries for each executed order.
        """
        # Run a few evaluation episodes (no learning)
        orders = []
        order_idx = 0
        for ep in range(num_episodes):
            obs, info = self.reset()
            info['episode'] = ep
            info['order_idx'] = order_idx
            done = False
            truncated = False
            order_info = [info]

            step = 0
            while not truncated:
                # Use neutral action 0 after order completion to keep collecting market data
                if done:
                    action = 0
                else:
                    action, _ = model.predict(obs, deterministic=True)

                obs, reward, done, truncated, info = self.step(action)
                # Annotate step info with episode and order index
                info['episode'] = ep
                info['order_idx'] = order_idx
                info['current_step'] = step
                order_info.append(info)
                step += 1

                self.render()  # print debug line each step
            order_idx += 1
            orders.append(order_info)
            logger.debug(f"Episode {ep} ended. Final Cost: {info['total_reward']:.2f}, Truncated: {truncated}, Info: {info}")

        self.close()

        return orders

