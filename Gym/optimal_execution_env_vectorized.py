import gymnasium as gym
from gymnasium import spaces
import numpy as np
import torch
import logging 
from typing import Optional, Tuple, Dict, Any, List
import platform
import pandas as pd
from stable_baselines3.common.vec_env import VecEnv
from gpu_utils import get_torch_device, set_random_seed

# Set up logger for this module
logger = logging.getLogger(__name__)

class VectorizedMultiOrderExecutionEnv(VecEnv):
    """
    Vectorized version of MultiOrderExecutionEnv that can run multiple environments in parallel.
    This is a true vectorized environment that maintains separate state for each parallel environment
    while leveraging GPU acceleration for tensor operations.
    """
    metadata = {"render_modes": ["human"]}

    def __init__(self, stock_df_list, orders_df, impact_coef, decay_rate, num_envs, 
                 min_rate=0.0, max_rate=0.1, window_size=1, unfilled_penalty=1e6, 
                 device: Optional[str] = None, render_mode=None, seed=None):
        """
        Initialize the Vectorized Optimal Execution Environment.
        
        @param stock_df_list: List of DataFrames containing stock data
        @param orders_df: DataFrame containing orders (may include 'Y' and 'tau' columns for per-order parameters)
        @param impact_coef: Immediate impact coefficient (γ) - used as fallback if orders_df lacks 'Y' column
        @param decay_rate: Residual decay factor (κ) - used as fallback if orders_df lacks 'tau' column
        @param num_envs: Number of parallel environments
        @param min_rate: Minimum fraction of volume that can be traded
        @param max_rate: Maximum fraction of volume that can be traded
        @param window_size: Number of steps to consider for residual impact decay
        @param unfilled_penalty: Penalty for unfilled orders
        @param device: Device to use for tensor operations
        @param render_mode: Render mode for the environment
        @param seed: Random seed for reproducibility
        """
        # Device selection: explicit > MPS > CUDA > CPU
        # Force CPU for compatibility with training environment unless explicitly specified
        if device is None:
            self.device = "cpu"
            logger.info(f"Using device: {self.device} (forced for compatibility)")
        else:
            self.device = device
            logger.info(f"Using device: {self.device} (explicitly set)")

        # Set the seed for reproducibility
        self._seed = seed if seed is not None else 0
        self._set_seed(self._seed)
        self._np_random = np.random.RandomState(self._seed)
        

        # Define observation space
        # Dimensions are: 
        # [mid_price, volume, time_remaining, shares_remaining, adv_pct, ehv_pct,
        #  signal, last_fill_price, last_trade_size, immediate_impact, 
        #  accumulated_impact,arrival_price, regime, daily_vol_lag1, daily_vol_5d]
        obs_dim = 15
        obs_low = np.array([
            0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -1.0, 0.0, 0.0, 0.0, 0.0, 0.0, -1.0, 0.0, 0.0
        ], dtype=np.float32)
        obs_high = np.array([
            1e10, 1e16, 600.0, 1e16, 1.0, 1.0, 1.0, 1e10, 1e16, 1e10, 1e10, 1e10, 1.0, 1e10, 1e10
        ], dtype=np.float32)
        observation_space = spaces.Box(low=obs_low, high=obs_high, shape=(obs_dim,), dtype=np.float32)
        
        # Define action space (same as original)
        action_space = spaces.Discrete(11)
        
        # Initialize VecEnv
        super().__init__(num_envs, observation_space, action_space)
        
        self.render_mode = render_mode

        # Save parameters
        self.stock_df_list = stock_df_list
        self.orders_df = orders_df
        self.impact_coef = torch.tensor(impact_coef, device=self.device, dtype=torch.float32)
        self.decay_rate = torch.tensor(decay_rate, device=self.device, dtype=torch.float32)
        self.window_size = window_size
        self.min_rate = min_rate
        self.max_rate = max_rate
        self.unfilled_penalty = unfilled_penalty
        
        # Action values tensor
        self.action_values = torch.tensor(
            [-1, -0.75, -0.5, -0.25, -0.1, 0.0, 0.1, 0.25, 0.5, 0.75, 1], 
            dtype=torch.float32, device=self.device
        )

        # Initialize vectorized state variables
        self._init_vectorized_state()
        
        # Preload data arrays for fast access
        self._preload_data_arrays()

    def _set_seed(self, seed):
        """
        Set the random seed for reproducibility.
        @param seed: Seed value to set
        """
        np.random.seed(seed)
        set_random_seed(seed)
        
    def _init_vectorized_state(self):
        """
        Initialize all state variables for the vectorized environment.
        This includes environment state, order details, and trading state.
        """
        # Environment state
        self.current_step = torch.zeros(self.num_envs, device=self.device, dtype=torch.int32)
        self.shares_remaining = torch.zeros(self.num_envs, device=self.device, dtype=torch.int64)
        self.arrival_price = torch.zeros(self.num_envs, device=self.device, dtype=torch.float32)
        self.accumulated_impact = torch.zeros(self.num_envs, device=self.device, dtype=torch.float32)
        self.done = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        
        # Order details
        self.order_idx = torch.zeros(self.num_envs, device=self.device, dtype=torch.int64)
        self.tickers = [None] * self.num_envs
        self.order_qty = torch.zeros(self.num_envs, device=self.device, dtype=torch.int64)
        self.adv_pct = torch.zeros(self.num_envs, device=self.device, dtype=torch.float32)
        self.ehv_pct = torch.zeros(self.num_envs, device=self.device, dtype=torch.float32)
        self.adv = torch.zeros(self.num_envs, device=self.device, dtype=torch.float32)
        self.start_time = torch.zeros(self.num_envs, device=self.device, dtype=torch.int64)
        self.end_time = torch.zeros(self.num_envs, device=self.device, dtype=torch.int64)
        self.time_horizon = torch.zeros(self.num_envs, device=self.device, dtype=torch.int64)
        self.side = torch.zeros(self.num_envs, device=self.device, dtype=torch.int8)
        
        # Trading state
        self.immediate_impact = torch.zeros(self.num_envs, device=self.device, dtype=torch.float32)
        self.order_vwap = torch.zeros(self.num_envs, device=self.device, dtype=torch.float32)
        self.last_fill_price = torch.zeros(self.num_envs, device=self.device, dtype=torch.float32)
        self.last_trade_size = torch.zeros(self.num_envs, device=self.device, dtype=torch.int64)
        self.last_action_fraction = torch.zeros(self.num_envs, device=self.device, dtype=torch.float32)
        self.last_trade_step = torch.full((self.num_envs,), -1, device=self.device, dtype=torch.int32)
        self.total_market_volume = torch.zeros(self.num_envs, device=self.device, dtype=torch.float32)
        self.total_cost = torch.zeros(self.num_envs, device=self.device, dtype=torch.float32)
        
        # Environment-specific propagator parameters (initialized with defaults, will be set in reset)
        self.env_impact_coef = torch.zeros(self.num_envs, device=self.device, dtype=torch.float32)
        self.env_decay_rate = torch.zeros(self.num_envs, device=self.device, dtype=torch.float32)

    def _preload_data_arrays(self):
        """
        Convert all stock dataframes into tensors for faster access.
        """
        self.data_arrays = {}
        for ticker, df_raw in self.stock_df_list.items():
            if isinstance(df_raw.columns, pd.MultiIndex):
                df = df_raw.xs(ticker, axis=1, level=1)
            else:
                df = df_raw
        
            # Helper function to handle NaN values
            def safe_tensor(data, default_value=0.0):
                arr = data.to_numpy(dtype=np.float32) if hasattr(data, 'to_numpy') else np.array(data, dtype=np.float32)
                arr = np.nan_to_num(arr, nan=default_value)
                return torch.tensor(arr, device=self.device)
        
            # Calculate VWAP with NaN protection
            if 'vwap' in df.columns:
                vwap_values = df['vwap'].to_numpy(dtype=np.float32)
            else:
                # Calculate from OHLC
                vwap_values = ((df['trade_high'] + df['trade_low'] + df['trade_last']) / 3.0).to_numpy(dtype=np.float32)
        
            # Replace NaN VWAP values with mid price
            mid_prices = ((df['trade_high'] + df['trade_low']) / 2.0).to_numpy(dtype=np.float32)
            vwap_values = np.where(np.isnan(vwap_values), mid_prices, vwap_values)
        
            self.data_arrays[ticker] = {
                'trade_high': safe_tensor(df['trade_high']),
                'trade_low': safe_tensor(df['trade_low']),
                'trade_last': safe_tensor(df['trade_last']),
                'trade_volume': safe_tensor(df['trade_volume']),
                'vwap': torch.tensor(vwap_values, device=self.device),
                'Signal': safe_tensor(df['Signal'] if 'Signal' in df.columns else pd.Series(np.zeros(len(df)))),
                'Regime': safe_tensor(df['Regime'] if 'Regime' in df.columns else pd.Series(np.zeros(len(df)))),
                'daily_volatility': safe_tensor(
                    df['daily_volatility'] if 'daily_volatility' in df.columns else pd.Series(np.full(len(df), 0.02)),
                    default_value=0.02
                ),
                'daily_volatility_lag1': safe_tensor(
                    df['daily_volatility_lag1'] if 'daily_volatility_lag1' in df.columns else pd.Series(np.full(len(df), 0.02)),
                    default_value=0.02
                ),
                'daily_volatility_5d': safe_tensor(
                    df['daily_volatility_5d'] if 'daily_volatility_5d' in df.columns else pd.Series(np.full(len(df), 0.02)),
                    default_value=0.02
                ),
                'dates': pd.to_datetime(df.index) if not isinstance(df.index, pd.DatetimeIndex) else df.index
            }


    def reset(self, seed=None, options=None):
        """
        Reset all environments and return initial observations.
        @param seed: Optional seed for random number generation
        @param options: Optional reset options (not used)
        @return: Initial observations
        """
        # Handle seeding according to gymnasium standards
        if seed is not None:
            self._seed = seed
            self._set_seed(seed)
            self._np_random = np.random.RandomState(seed)
        
        # Initialize _np_random if not already done (required by gymnasium)
        if not hasattr(self, '_np_random') or self._np_random is None:
            self._np_random = np.random.RandomState(self._seed if hasattr(self, '_seed') else None)
        
        # Pick random batch of order indices using seeded random
        num_orders = len(self.orders_df)
        order_indices = self._np_random.randint(0, num_orders, size=self.num_envs)
        
        self.order_idx = torch.tensor(order_indices, device=self.device)
        order_rows = self.orders_df.iloc[order_indices]

        # Extract order details for the batch
        self.tickers = order_rows['ticker'].tolist()
        self.order_qty = torch.tensor(order_rows['order_qty'].values, device=self.device, dtype=torch.int64)
        self.adv_pct = torch.tensor(order_rows['adv_pct'].values, device=self.device, dtype=torch.float32)
        self.ehv_pct = torch.tensor(order_rows['ehv_pct'].values, device=self.device, dtype=torch.float32)
        
        # Extract adv values robustly
        adv_list = []
        for x in order_rows['adv']:
            if isinstance(x, pd.Series):
                adv_list.append(float(x.iloc[0]))
            elif isinstance(x, (list, tuple, np.ndarray)):
                adv_list.append(float(x[0]))
            else:
                adv_list.append(float(x))
        self.adv = torch.tensor(adv_list, device=self.device, dtype=torch.float32)
        
        self.start_time = torch.tensor(order_rows['start_time'].values, device=self.device, dtype=torch.int64)
        self.end_time = torch.tensor(order_rows['end_time'].values, device=self.device, dtype=torch.int64)
        self.time_horizon = torch.tensor(order_rows['time_horizon'].values, device=self.device, dtype=torch.int64)
        self.side = torch.tensor([1 if s.lower() == 'buy' else -1 for s in order_rows['side']], 
                                device=self.device, dtype=torch.int8)

        # Set environment-specific impact parameters from orders_df or use defaults
        if 'Y' in order_rows.columns:
            Y_values = order_rows['Y'].values
            # Check for NaN values and replace with median
            nan_mask = pd.isna(Y_values)
            if nan_mask.any():
                valid_values = Y_values[~nan_mask]
                if len(valid_values) > 0:
                    median_value = np.median(valid_values)
                    logger.warning(f"Found {nan_mask.sum()} NaN Y values, replacing with median {median_value}")
                    Y_values = np.where(nan_mask, median_value, Y_values)
                else:
                    logger.warning(f"All Y values are NaN, using default {self.impact_coef.item()}")
                    Y_values = np.full_like(Y_values, self.impact_coef.item())
            self.env_impact_coef = torch.tensor(Y_values, device=self.device, dtype=torch.float32)
        else:
            # Use default impact coefficient for all environments
            self.env_impact_coef = self.impact_coef.expand(self.num_envs)
            
        if 'tau' in order_rows.columns:
            tau_values = order_rows['tau'].values
            # Check for NaN values and replace with median
            nan_mask = pd.isna(tau_values)
            if nan_mask.any():
                valid_values = tau_values[~nan_mask]
                if len(valid_values) > 0:
                    median_value = np.median(valid_values)
                    logger.warning(f"Found {nan_mask.sum()} NaN tau values, replacing with median {median_value}")
                    tau_values = np.where(nan_mask, median_value, tau_values)
                else:
                    logger.warning(f"All tau values are NaN, using default {self.decay_rate.item()}")
                    tau_values = np.full_like(tau_values, self.decay_rate.item())
            self.env_decay_rate = torch.tensor(tau_values, device=self.device, dtype=torch.float32)
        else:
            # Use default decay rate for all environments
            self.env_decay_rate = self.decay_rate.expand(self.num_envs)

        # Handle dates for global indexing
        if 'date' in order_rows.columns:
            self.order_dates = order_rows['date'].tolist()
            self.global_start_indices = torch.zeros(self.num_envs, device=self.device, dtype=torch.int64)
            
            for i, (ticker, date) in enumerate(zip(self.tickers, self.order_dates)):
                ticker_dates = self.data_arrays[ticker]['dates']
                
                if isinstance(date, str):
                    order_date = pd.to_datetime(date)
                else:
                    order_date = pd.to_datetime(date)
                
                if isinstance(ticker_dates, pd.DatetimeIndex):
                    normalized_ticker_dates = ticker_dates.normalize()
                else:
                    # Convert to DatetimeIndex first if it's a regular Index
                    ticker_dates = pd.to_datetime(ticker_dates)
                    normalized_ticker_dates = ticker_dates.normalize()
            
                # normalize for date only comparison
                normalized_order_date = pd.Timestamp(order_date).normalize()                
            
                # Find matching date
                date_mask = normalized_ticker_dates == normalized_order_date
                date_indices = np.where(date_mask)[0]
                
                if len(date_indices) == 0:
                    # Try finding the nearest date
                    date_diffs = abs(normalized_ticker_dates - normalized_order_date)
                    nearest_idx = date_diffs.argmin()
                    
                    # Check if nearest date is within acceptable range (e.g., 3 days)
                    if date_diffs.iloc[nearest_idx].days <= 3:
                        date_indices = np.array([nearest_idx])
                        logger.warning(f"No exact match for {ticker} on {order_date.date()}, "
                                     f"using nearest date: {ticker_dates[nearest_idx].date()}")
                    else:
                        raise ValueError(f"No data found for ticker {ticker} within 3 days of {order_date.date()}")
                
                self.global_start_indices[i] = date_indices[0] + self.start_time[i]
        else:
            # If no date column in orders, just use start_time as indices
            self.global_start_indices = self.start_time.clone()
            self.order_dates = None

        # Reset state variables
        self.current_step.zero_()
        self.shares_remaining = self.order_qty.clone()
        self.accumulated_impact.zero_()
        self.done.zero_()
        self.last_fill_price.zero_()
        self.last_trade_size.zero_()
        self.last_action_fraction.zero_()
        self.immediate_impact.zero_()
        self.total_market_volume.zero_()
        self.total_cost.zero_()
        self.last_trade_step.fill_(-1)
        self.order_vwap.zero_()

        # Compute arrival prices
        self.arrival_price.zero_()
        for ticker in set(self.tickers):
            env_indices = [i for i, t in enumerate(self.tickers) if t == ticker]
            env_indices_tensor = torch.tensor(env_indices, device=self.device, dtype=torch.long)
            
            start_indices = self.global_start_indices[env_indices_tensor]
            
            # Data validation
            max_len = len(self.data_arrays[ticker]['trade_high'])
            if (start_indices < 0).any() or (start_indices >= max_len).any():
                for idx, env_idx in enumerate(env_indices):
                    if self.global_start_indices[env_idx] < 0 or self.global_start_indices[env_idx] >= max_len:
                        # Clamp to valid range
                        self.global_start_indices[env_idx] = torch.clamp(
                            self.global_start_indices[env_idx], 
                            0, 
                            max_len - 1
                        )
                        logger.warning(f"Clamped index for order {self.order_idx[env_idx]} (ticker={ticker})")
                
                # Re-get indices after clamping
                start_indices = self.global_start_indices[env_indices_tensor]

            highs = self.data_arrays[ticker]['trade_high'][start_indices]
            lows = self.data_arrays[ticker]['trade_low'][start_indices]
            self.arrival_price[env_indices_tensor] = (highs + lows) / 2.0

        # Initialize first step values
        market_data = self._get_market_data_batch(['trade_volume', 'vwap', 'trade_high', 'trade_low', 'daily_volatility_lag1'], use_prior_step=False)
        current_volumes = market_data['trade_volume']
        vwap_prices = market_data['vwap']
        
        # Initialize last_fill_price to 0.0 since no trades have occurred yet
        self.last_fill_price.zero_()
        self.last_trade_size.zero_()
        self.immediate_impact.zero_()
        self.accumulated_impact.zero_()
        self.order_vwap.zero_()

        # Return initial observations
        obs = self._get_observation()

        if not np.isfinite(obs).all():
            idx = np.where(~np.isfinite(obs))[0]
            raise RuntimeError(f"reset: Env returned non-finite features at idx={idx}, values={obs[idx]}")

        return obs

    def step_async(self, actions):
        """
        Store actions for async step.
        @param actions: List of actions for each environment
        """
        self.actions = torch.tensor(actions, device=self.device, dtype=torch.int64)

    def _calculate_deterministic_impact(self, trade_sizes):
        """
        Calculates the deterministic, immediate impact of a trade.

        Args:
            trade_sizes: Trade sizes for each environment

        Returns:
            Deterministic impact for each environment
        """
        # TODO: use a volume profile for this to scale for intraday seasonality
        minute_volume = self.adv / 390.0
        epsilon = trade_sizes.float() / minute_volume
        deterministic_impact = self.env_impact_coef * epsilon
        
        return deterministic_impact




    def step_wait(self):
        """
        Execute the stored actions and return results.
        @return: Tuple of (observations, rewards, dones, infos)
        """
        # Validate actions
        if not (0 <= self.actions).all() or not (self.actions < len(self.action_values)).all():
            raise ValueError(f"Invalid actions {self.actions}. Must be between 0 and {len(self.action_values)-1}")

        # Get current market data
        market_data = self._get_market_data_batch(['trade_volume', 'vwap', 'trade_high', 'trade_low', 'daily_volatility_lag1'], use_prior_step=False)
        current_volumes = market_data['trade_volume']
        vwap_prices = market_data['vwap']
        mid_prices = (market_data['trade_high'] + market_data['trade_low']) * 0.5
        daily_sigma = market_data['daily_volatility_lag1']

        # Compute trade sizes
        fractions = self.action_values[self.actions]
        trade_sizes = torch.where(
            self.shares_remaining > 0, 
            torch.min(self.ehv_pct * (1 + fractions) * current_volumes, self.shares_remaining.float()),
            torch.zeros_like(self.shares_remaining, dtype=torch.float32)
        )
        trade_sizes = torch.round(trade_sizes).to(torch.int64)

        # Update market volume
        self.total_market_volume += current_volumes + trade_sizes.float()
        self.total_market_volume = torch.clamp(self.total_market_volume, min=1)

        # Compute impacts using environment-specific parameters
        deterministic_impact = self._calculate_deterministic_impact(trade_sizes)
        self.immediate_impact = deterministic_impact
        
        # Residual impact decay using environment-specific parameters
        delta_t = torch.where(
            self.last_trade_step >= 0,
            (self.current_step - self.last_trade_step).float(),
            torch.ones_like(self.current_step, dtype=torch.float32)
        )
        lambda_decay = torch.exp(-delta_t / self.env_decay_rate)
        self.accumulated_impact = lambda_decay * self.accumulated_impact + deterministic_impact
        fill_prices = vwap_prices * (1 + self.side * self.accumulated_impact)

        minutely_sigma = daily_sigma / np.sqrt(390.0)
        stochastic_noise = torch.normal(mean=0.0, std=minutely_sigma)
        total_log_return = deterministic_impact + stochastic_noise

        # Update order VWAP
        prior_filled_qty = self.order_qty - self.shares_remaining
        total_filled_qty = prior_filled_qty + trade_sizes
        
        trade_mask = trade_sizes > 0
        first_fill_mask = (prior_filled_qty == 0) & trade_mask
        additional_fill_mask = (prior_filled_qty > 0) & trade_mask

        new_order_vwap = self.order_vwap.clone()
        new_order_vwap = torch.where(first_fill_mask, fill_prices, new_order_vwap)
        new_order_vwap = torch.where(
            additional_fill_mask,
            (self.order_vwap * prior_filled_qty.float() + fill_prices * trade_sizes.float()) / total_filled_qty.float(),
            new_order_vwap
        )
        self.order_vwap = torch.where(trade_mask, new_order_vwap, self.order_vwap)

        # Update state
        prev_shares = self.shares_remaining.clone()
        self.shares_remaining = torch.max(prev_shares - trade_sizes, torch.zeros_like(prev_shares))

        # Update last trade step
        trade_executed_mask = trade_sizes > 0
        self.last_trade_step[trade_executed_mask] = self.current_step[trade_executed_mask]

        # Advance time
        self.current_step += 1
        
        # Only update last_fill_price when there's actually a trade
        self.last_fill_price = torch.where(trade_executed_mask, fill_prices, self.last_fill_price)
        self.last_trade_size = trade_sizes
        self.last_action_fraction = fractions

        # Compute rewards
        terminated_mask = self.shares_remaining <= 0
        truncated_mask = self.current_step >= self.time_horizon
        
        reward = torch.zeros(self.num_envs, device=self.device)
        
        # Calculate slippage
        price_performance = (fill_prices - self.arrival_price) / self.arrival_price
        slippage = -self.side * price_performance * (trade_sizes / self.order_qty.float())
        trade_cost = slippage

        # Rate deviation penalty
        target_completion_ratio = self.current_step.float() / self.time_horizon.float()
        actual_completion_ratio = (self.order_qty - self.shares_remaining).float() / self.order_qty.float()
        rate_deviation = actual_completion_ratio - target_completion_ratio
        rate_penalty_coef = 0.001
        rate_penalty = -rate_penalty_coef * torch.abs(rate_deviation) * (trade_sizes / self.order_qty.float())

        # Apply unfilled penalty
        if truncated_mask.any():
            unfilled_ratio = self.shares_remaining[truncated_mask] / self.order_qty[truncated_mask]
            reward[truncated_mask] += -self.unfilled_penalty * unfilled_ratio

        reward = trade_cost + rate_penalty
        self.total_cost += reward

        # Update done flags
        self.done = truncated_mask

        # Get new observations
        obs = self._get_observation()

        if not np.isfinite(obs).all():
            idx = np.where(~np.isfinite(obs))[0]
            raise RuntimeError(f"step_asynch:Env returned non-finite features at idx={idx}, values={obs[idx]}")

        
        # Convert to numpy for VecEnv interface
        # Note: This is not strictly necessary, but keeps interface consistent
        obs_np = obs
        rewards_np = reward.cpu().numpy()
        dones_np = self.done.cpu().numpy()
        # Create empty infos for each environment
        # TODO: Populate with relevant info if needed
        infos = [{} for _ in range(self.num_envs)]

        return obs_np, rewards_np, dones_np, infos

    def _get_market_data_batch(self, fields: List[str], use_prior_step: bool = False) -> Dict[str, torch.Tensor]:
        """
        Gather market data for current step for specified fields.
        @param fields: List of fields to gather (e.g., ['trade_high', 'trade_low', 'trade_volume'])
        @param use_prior_step: Whether to use data from the prior step (default False)
        @return: Dictionary of tensors for each requested field
        """
        indices = self.global_start_indices + self.current_step
        if use_prior_step:
            indices = indices - 1

        results = {field: torch.zeros(self.num_envs, device=self.device, dtype=torch.float32) for field in fields}
        
        for ticker in set(self.tickers):
            # Skip invalid tickers or ones not present in data
            if ticker is None or ticker not in self.data_arrays:
                continue

            env_indices = [i for i, t in enumerate(self.tickers) if t == ticker]
            if not env_indices:
                continue
                
            env_indices_tensor = torch.tensor(env_indices, device=self.device, dtype=torch.long)
            market_indices = indices[env_indices_tensor]

            for field in fields:
                if field in self.data_arrays[ticker]:
                    data_tensor = self.data_arrays[ticker][field]
                    # Add bounds checking to prevent index out of bounds
                    valid_indices = torch.clamp(market_indices, 0, len(data_tensor) - 1)
                    results[field][env_indices_tensor] = data_tensor[valid_indices]

        # Ensure all results are on the same device as the environment
        for field in results:
            results[field] = results[field].to(self.device)
        
        return results

    def _get_observation(self):
        """
        Return batch of observations.
        @return: Numpy array of observations for all environments
        """
        market_data = self._get_market_data_batch(['trade_high', 'trade_low', 'trade_volume', 'Signal', 'Regime', 'daily_volatility_lag1', 'daily_volatility_5d'], use_prior_step=True)
        
        mid_price = (market_data['trade_high'] + market_data['trade_low']) * 0.5
        volume = market_data['trade_volume']
        signal = market_data['Signal']
        regime = market_data['Regime']
        vol_lag1 = market_data['daily_volatility_lag1']
        vol5d = market_data['daily_volatility_5d']

        obs_tensor = torch.stack([
            mid_price,
            volume,
            (self.time_horizon - self.current_step).float(),
            self.shares_remaining.float(),
            self.adv_pct,
            self.ehv_pct,
            signal,
            self.last_fill_price,
            self.last_trade_size.float(),
            self.immediate_impact,
            self.accumulated_impact,
            self.arrival_price,
            regime,
            vol_lag1,
            vol5d
        ], dim=1)

        return obs_tensor.cpu().numpy()

    def render(self, mode='human'):
        """Render the environment (simplified for vectorized version)."""
        if mode == 'human':
            print(f"Vectorized Env: Step {self.current_step[0].item()}, "
                  f"Shares remaining: {self.shares_remaining[0].item()}")

    def close(self):
        """
        Close the environment.
        """
        pass

    def seed(self, seed=None):
        """
        Set the seed for the environment.
        @param seed: Seed value to set (optional)
        
        WARNING: This method is deprecated. Use env.reset(seed=seed) instead.
        """
        import warnings
        warnings.warn(
            "The 'seed' method is deprecated. Use 'env.reset(seed=seed)' instead.",
            DeprecationWarning,
            stacklevel=2
        )
        
        if seed is not None:
            self._seed = seed
            self._set_seed(seed)
            self._np_random = np.random.RandomState(seed)
        return [seed] * self.num_envs

    def execute_orders(self, model, num_episodes=10, fixed_order_indices=None):
        """
        Execute orders using the provided model (maintains same interface as original).
        @param model: The trained RL model to use for action selection.
        @param num_episodes: Number of orders to execute in this run.
        @param fixed_order_indices: Optional list of specific order indices to use for consistent comparison.
                                  If None, random orders will be selected. If provided, num_episodes will be
                                  set to len(fixed_order_indices).
        @return: Tuple of (orders_list, order_indices_used) where orders_list contains execution results
                and order_indices_used contains the actual order indices that were executed.
        """
        orders = []
        
        # If fixed indices provided, use them; otherwise generate new ones
        if fixed_order_indices is not None:
            num_episodes = len(fixed_order_indices)
            order_indices_used = fixed_order_indices.copy()
        else:
            # Generate random order indices once and reuse them
            num_orders = len(self.orders_df)
            if not hasattr(self, '_np_random') or self._np_random is None:
                self._np_random = np.random.RandomState(self._seed)
            order_indices_used = self._np_random.choice(num_orders, size=num_episodes, replace=False).tolist()
        
        for ep in range(num_episodes):
            # Use specific order index for this episode
            obs = self._reset_with_order_index(order_indices_used[ep])
            done = np.zeros(self.num_envs, dtype=bool)
            episode_data = []
            
            # For compatibility, we'll track the first environment
            env_idx = 0
            order_info = []
            
            step = 0
            while not done[env_idx]:
                # Get action from model (for first environment)
                if hasattr(model, 'predict'):
                    action, _ = model.predict(obs[env_idx], deterministic=False)
                else:
                    # For baseline strategies
                    action, _ = model.predict(obs[env_idx])
                
                # Create actions for all environments (use same action for simplicity)
                actions = np.full(self.num_envs, action)
                
                # Step the vectorized environment
                self.step_async(actions)
                obs, rewards, dones, infos = self.step_wait()
                
                # Get current market data for the first environment
                market_data = self._get_market_data_batch(['trade_volume', 'vwap', 'trade_high', 'trade_low'], use_prior_step=False)
                current_trade_volume = market_data['trade_volume'][env_idx].item()
                vwap_price = market_data['vwap'][env_idx].item()
                mid_price = ((market_data['trade_high'][env_idx] + market_data['trade_low'][env_idx]) * 0.5).item()
                
                # Track info for first environment (for compatibility)
                info = {
                    'order_idx': self.order_idx[env_idx].item(),
                    'ticker': self.tickers[env_idx],
                    'order_qty': self.order_qty[env_idx].item(),
                    'adv_pct': self.adv_pct[env_idx].item(),
                    'ehv_pct': self.ehv_pct[env_idx].item(),
                    'start_time': self.start_time[env_idx].item(),
                    'end_time': self.end_time[env_idx].item(),
                    'time_horizon': self.time_horizon[env_idx].item(),
                    'side': 'buy' if self.side[env_idx].item() == 1 else 'sell',
                    'shares_remaining': self.shares_remaining[env_idx].item(),
                    'last_fill_price': self.last_fill_price[env_idx].item(),
                    'last_trade_size': self.last_trade_size[env_idx].item(),
                    'immediate_impact': self.immediate_impact[env_idx].item(),
                    'accumulated_impact': self.accumulated_impact[env_idx].item(),
                    'arrival_price': self.arrival_price[env_idx].item(),
                    'order_vwap': self.order_vwap[env_idx].item(),
                    'total_reward': self.total_cost[env_idx].item(),
                    'action_percentage': float(self.last_action_fraction[env_idx].item()),
                    'adv': self.adv[env_idx].item(),
                    'current_step': step,
                    'episode': ep,
                    'order_idx': ep,
                    # Add missing fields required by plotting functions
                    'mid_price': mid_price,
                    'vwap_price': vwap_price,
                    'current_trade_volume': current_trade_volume
                }
                
                if self.order_dates:
                    info['date'] = str(self.order_dates[env_idx])
                
                order_info.append(info)
                done = dones
                step += 1
            
            orders.append(order_info)
        
        return orders, order_indices_used

    def _reset_with_order_index(self, order_idx):
        """Reset environment with a specific order index."""
        # Initialize _np_random if not already done
        if not hasattr(self, '_np_random') or self._np_random is None:
            self._np_random = np.random.RandomState(self._seed)
        
        # Validate order_idx
        if order_idx < 0 or order_idx >= len(self.orders_df):
            raise ValueError(f"order_idx {order_idx} out of range [0, {len(self.orders_df)})")
        
        # Use the specified order index for the first environment, random for others
        order_indices = [order_idx] + self._np_random.randint(
            0, len(self.orders_df), size=self.num_envs-1).tolist()
        
        self.order_idx = torch.tensor(order_indices, device=self.device)
        order_rows = self.orders_df.iloc[order_indices]

        # Validate that we got valid data
        if order_rows.empty:
            raise ValueError(f"No order data found for indices {order_indices}")

        # Extract order details for the batch with validation
        self.tickers = order_rows['ticker'].tolist()
        if all(t is None for t in self.tickers):
            raise ValueError("All tickers are None - check orders_df data")
            
        self.order_qty = torch.tensor(order_rows['order_qty'].values, device=self.device, dtype=torch.int64)
        if torch.all(self.order_qty == 0):
            raise ValueError("All order quantities are 0 - check orders_df data")
            
        self.adv_pct = torch.tensor(order_rows['adv_pct'].values, device=self.device, dtype=torch.float32)
        self.ehv_pct = torch.tensor(order_rows['ehv_pct'].values, device=self.device, dtype=torch.float32)
        
        # Extract adv values robustly
        adv_list = []
        for x in order_rows['adv']:
            if isinstance(x, pd.Series):
                adv_list.append(float(x.iloc[0]))
            elif isinstance(x, (list, tuple, np.ndarray)):
                adv_list.append(float(x[0]))
            else:
                adv_list.append(float(x))
        self.adv = torch.tensor(adv_list, device=self.device, dtype=torch.float32)
        
        self.start_time = torch.tensor(order_rows['start_time'].values, device=self.device, dtype=torch.int64)
        self.end_time = torch.tensor(order_rows['end_time'].values, device=self.device, dtype=torch.int64)
        self.time_horizon = torch.tensor(order_rows['time_horizon'].values, device=self.device, dtype=torch.int64)
        self.side = torch.tensor([1 if s.lower() == 'buy' else -1 for s in order_rows['side']], 
                                device=self.device, dtype=torch.int8)

        # Set environment-specific impact parameters from orders_df or use defaults
        if 'Y' in order_rows.columns:
            Y_values = order_rows['Y'].values
            # Check for NaN values and replace with median
            nan_mask = pd.isna(Y_values)
            if nan_mask.any():
                valid_values = Y_values[~nan_mask]
                if len(valid_values) > 0:
                    median_value = np.median(valid_values)
                    logger.warning(f"Found {nan_mask.sum()} NaN Y values, replacing with median {median_value}")
                    Y_values = np.where(nan_mask, median_value, Y_values)
                else:
                    logger.warning(f"All Y values are NaN, using default {self.impact_coef.item()}")
                    Y_values = np.full_like(Y_values, self.impact_coef.item())
            self.env_impact_coef = torch.tensor(Y_values, device=self.device, dtype=torch.float32)
        else:
            # Use default impact coefficient for all environments
            self.env_impact_coef = self.impact_coef.expand(self.num_envs)
            
        if 'tau' in order_rows.columns:
            tau_values = order_rows['tau'].values
            # Check for NaN values and replace with median
            nan_mask = pd.isna(tau_values)
            if nan_mask.any():
                valid_values = tau_values[~nan_mask]
                if len(valid_values) > 0:
                    median_value = np.median(valid_values)
                    logger.warning(f"Found {nan_mask.sum()} NaN tau values, replacing with median {median_value}")
                    tau_values = np.where(nan_mask, median_value, tau_values)
                else:
                    logger.warning(f"All tau values are NaN, using default {self.decay_rate.item()}")
                    tau_values = np.full_like(tau_values, self.decay_rate.item())
            self.env_decay_rate = torch.tensor(tau_values, device=self.device, dtype=torch.float32)
        else:
            # Use default decay rate for all environments
            self.env_decay_rate = self.decay_rate.expand(self.num_envs)

        # Handle dates for global indexing
        if 'date' in order_rows.columns:
            self.order_dates = order_rows['date'].tolist()
            self.global_start_indices = torch.zeros(self.num_envs, device=self.device, dtype=torch.int64)
            
            for i, (ticker, date) in enumerate(zip(self.tickers, self.order_dates)):
                ticker_dates = self.data_arrays[ticker]['dates']
                
                if isinstance(date, str):
                    order_date = pd.to_datetime(date)
                else:
                    order_date = pd.to_datetime(date)
                
                # Normalize times for date-only comparison
                normalized_ticker_dates = ticker_dates.normalize()
                normalized_order_date = pd.Timestamp(order_date).normalize()
                
                # Find matching date
                date_mask = normalized_ticker_dates == normalized_order_date
                date_indices = np.where(date_mask)[0]
                
                if len(date_indices) == 0:
                    # Try finding the nearest date
                    date_diffs = abs(normalized_ticker_dates - normalized_order_date)
                    nearest_idx = date_diffs.argmin()
                    
                    # Check if nearest date is within acceptable range
                    if date_diffs.iloc[nearest_idx].days <= 3:
                        date_indices = np.array([nearest_idx])
                        logger.warning(f"No exact match for {ticker} on {order_date.date()}, "
                                     f"using nearest date: {ticker_dates[nearest_idx].date()}")
                    else:
                        raise ValueError(f"No data found for ticker {ticker} within 3 days of {order_date.date()}")
                
                self.global_start_indices[i] = date_indices[0] + self.start_time[i]
        else:
            self.order_dates = None
            self.global_start_indices = self.start_time.clone()

        # Reset state variables (but keep order data we just set)
        self.current_step = torch.zeros(self.num_envs, device=self.device, dtype=torch.int32)
        self.shares_remaining = self.order_qty.clone()  # Start with full order quantity
        self.accumulated_impact = torch.zeros(self.num_envs, device=self.device, dtype=torch.float32)
        self.done = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        
        # Trading state
        self.immediate_impact = torch.zeros(self.num_envs, device=self.device, dtype=torch.float32)
        self.order_vwap = torch.zeros(self.num_envs, device=self.device, dtype=torch.float32)
        self.last_fill_price = torch.zeros(self.num_envs, device=self.device, dtype=torch.float32)
        self.last_trade_size = torch.zeros(self.num_envs, device=self.device, dtype=torch.int64)
        self.last_action_fraction = torch.zeros(self.num_envs, device=self.device, dtype=torch.float32)
        self.last_trade_step = torch.full((self.num_envs,), -1, device=self.device, dtype=torch.int32)
        self.total_market_volume = torch.zeros(self.num_envs, device=self.device, dtype=torch.float32)
        self.total_cost = torch.zeros(self.num_envs, device=self.device, dtype=torch.float32)
        
        # Compute arrival prices
        self.arrival_price.zero_()
        for ticker in set(self.tickers):
            if ticker is None or ticker not in self.data_arrays:
                continue
                
            env_indices = [i for i, t in enumerate(self.tickers) if t == ticker]
            if not env_indices:
                continue
                
            env_indices_tensor = torch.tensor(env_indices, device=self.device, dtype=torch.long)
            start_indices = self.global_start_indices[env_indices_tensor]
            
            # Data validation
            max_len = len(self.data_arrays[ticker]['trade_high'])
            valid_indices = torch.clamp(start_indices, 0, max_len - 1)
            
            highs = self.data_arrays[ticker]['trade_high'][valid_indices]
            lows = self.data_arrays[ticker]['trade_low'][valid_indices]
            self.arrival_price[env_indices_tensor] = (highs + lows) / 2.0
        
        # Return initial observation
        return self._get_observation()

    def execute_fixed_orders(self, model, order_indices, return_indices=False):
        """
        Execute specific orders by their indices to ensure consistent evaluation across models.
        
        @param model: The trained RL model to use for action selection.
        @param order_indices: List of specific order indices to execute.
        @param return_indices: If True, return the order indices used (for compatibility).
        @return: List of order information dictionaries or tuple (orders, indices) if return_indices=True.
        """
        orders, used_indices = self.execute_orders(model, fixed_order_indices=order_indices)
        
        if return_indices:
            return orders, used_indices
        return orders

    # Required VecEnv abstract methods
    def env_is_wrapped(self, wrapper_class, indices=None):
        """
        Check if environments are wrapped with a given wrapper.
        @param wrapper_class: Class of the wrapper to check
        @param indices: Optional indices to check specific environments
        """
        return [False] * self.num_envs

    def env_method(self, method_name, *method_args, indices=None, **method_kwargs):
        """
        Call instance methods of vectorized environments.
        @param method_name: Name of the method to call
        @param method_args: Positional arguments for the method
        @param indices: Optional indices to call method on specific environments
        @param method_kwargs: Keyword arguments for the method
        @return: List of results from each environment
        """
        # For simplicity, we don't support arbitrary method calls
        return [None] * self.num_envs

    def get_attr(self, attr_name, indices=None):
        """
        Get attribute from vectorized environments.
        @param attr_name: Name of the attribute to get
        @param indices: Optional indices to get attribute from specific environments
        @return: List of attribute values for each environment
        """
        if hasattr(self, attr_name):
            attr_value = getattr(self, attr_name)
            return [attr_value] * self.num_envs
        return [None] * self.num_envs

    def set_attr(self, attr_name, value, indices=None):
        """
        Set attribute in vectorized environments.
        @param attr_name: Name of the attribute to set
        @param value: Value to set the attribute to
        @param indices: Optional indices to set attribute in specific environments
        """
        if hasattr(self, attr_name):
            setattr(self, attr_name, value)