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

# Constants
MINUTES_PER_DAY = 390.0
POV_CAP = 10.0 # Cap minutely POV at 10x volume

# TODO: integrate minutely volume and volatility profiles 


class VectorizedMultiOrderExecutionEnv(VecEnv):
    """
    Vectorized version of MultiOrderExecutionEnv that can run multiple environments in parallel.
    This is a true vectorized environment that maintains separate state for each parallel environment
    while leveraging GPU acceleration for tensor operations.
    """
    metadata = {"render_modes": ["human"]}

    def __init__(self, stock_df_list, orders_df, impact_coef, decay_rate, num_envs, 
                 min_rate=0.0, max_rate=0.1, risk_lambda=0.0, window_size=1, unfilled_penalty=1,
                 device: Optional[str] = None, render_mode=None, seed=None,
                 cost_model=None):
        """
        Initialize the Vectorized Optimal Execution Environment.
        
        Args:
            stock_df_list: List of DataFrames containing stock data
            orders_df: DataFrame containing orders (may include 'Y' and 'tau' columns for per-order parameters)
            impact_coef: Immediate impact coefficient (γ) - used as fallback if orders_df lacks 'Y' column
            decay_rate: Residual decay factor (κ) - used as fallback if orders_df lacks 'tau' column
            num_envs: Number of parallel environments
            min_rate: Minimum fraction of volume that can be traded 
            max_rate: Maximum fraction of volume that can be traded
            risk_lambda: Risk penalty factor for risk-adjusted returns (default: 0.0)
            window_size: Number of steps to consider for residual impact decay
            unfilled_penalty: Penalty for unfilled orders
            device: Device to use for tensor operations
            render_mode: Render mode for the environment
            seed: Random seed for reproducibility

        Returns:
            None
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
        # Optional pluggable cost model. If provided, environment will delegate cost computation.
        self.cost_model = cost_model
        
        # Action values tensor
        self.action_values = torch.tensor(
            [-1, -0.75, -0.5, -0.25, -0.1, 0.0, 0.1, 0.25, 0.5, 0.75, 1], 
            dtype=torch.float32, device=self.device
        )

        self.risk_lambda = torch.tensor(risk_lambda, device=self.device, dtype=torch.float32)

        # Initialize vectorized state variables
        self._init_vectorized_state()
        
        # Preload data arrays for fast access
        self._preload_data_arrays()

        self.logged_errors = {}

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
        # Cumulative market VWAP components over the order interval
        self.cum_market_volume = torch.zeros(self.num_envs, device=self.device, dtype=torch.float32)
        self.cum_market_dollars = torch.zeros(self.num_envs, device=self.device, dtype=torch.float32)
        
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
        
            # Replace non-finite VWAP values with mid price; fall back to last price then zero
            mid_prices = ((df['trade_high'] + df['trade_low']) / 2.0).to_numpy(dtype=np.float32)
            last_prices = df['trade_last'].to_numpy(dtype=np.float32)
            # Mid fallback: if mid is non-finite, use last; if last non-finite, use 0
            mid_safe = np.where(np.isfinite(mid_prices), mid_prices,
                                np.where(np.isfinite(last_prices), last_prices, 0.0))
            vwap_values = np.where(np.isfinite(vwap_values), vwap_values, mid_safe)
            vwap_values = np.nan_to_num(vwap_values, nan=0.0, posinf=0.0, neginf=0.0)
        
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
            # Check for NaN values or < 0 and replace with median
            nan_non_positive_mask = pd.isna(Y_values) | (Y_values < 0)
            if nan_non_positive_mask.any():
                valid_values = Y_values[~nan_non_positive_mask]
                if len(valid_values) > 0:
                    median_value = np.median(valid_values)
                    logger.warning(f"Found {nan_non_positive_mask.sum()} NaN or non-positive Y values, replacing with median {median_value}")
                    Y_values = np.where(nan_non_positive_mask, median_value, Y_values)
                else:
                    logger.warning(f"All Y values are NaN, using default {self.impact_coef.item()}")
                    Y_values = np.full_like(Y_values, self.impact_coef.item())
            self.env_impact_coef = torch.tensor(Y_values, device=self.device, dtype=torch.float32)
        else:
            # Use default impact coefficient for all environments
            self.env_impact_coef = self.impact_coef.expand(self.num_envs)
            
        if 'tau' in order_rows.columns:
            tau_values = order_rows['tau'].values
            # Check for NaN or non positiv evalues and replace with median
            nan_non_positive_mask = pd.isna(tau_values) | (tau_values < 0)
            if nan_non_positive_mask.any():
                valid_values = tau_values[~nan_non_positive_mask]
                if len(valid_values) > 0:
                    median_value = np.median(valid_values)
                    if 'tau_error' not in self.logged_errors:
                        logger.warning(f"Found {nan_non_positive_mask.sum()} NaN or non-positive tau values, replacing with median {median_value}")
                    tau_values = np.where(nan_non_positive_mask, median_value, tau_values)
                    self.logged_errors['tau_error'] = True
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
        self.cum_market_volume.zero_()
        self.cum_market_dollars.zero_()

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


    def _calculate_deterministic_impact(self, minute_pov):
        """
        Calculates the deterministic, immediate impact of a trade.

        Args:
            trade_sizes: Trade sizes for each environment
            current_volumes: Current volumes for each environment

        Returns:
            Deterministic impact for each environment
        """
        # Clip only positive PoVs into a safe range to avoid sqrt issues and overflow
        minute_pov_safe = torch.where(
            minute_pov > 0,
            torch.clamp(minute_pov, min=1e-6, max=POV_CAP),
            minute_pov
        )
        epsilon = self.side * torch.sqrt(minute_pov_safe.clamp_min(0.0))
        # Deterministic immediate term is Y * ε_t
        deterministic_impact = self.env_impact_coef * epsilon  
        
        return deterministic_impact

    def _calculate_trade_sizes(self, current_volumes, action_values, mode='DPOV'):
        """
        Calculates the trade sizes for each environment, this will aim to complete the order
        smoothly along the horizon and adjust pov based on prior mispredictions. Finally, it
        will adjus the pov based on the action values.

        Note: this could and should be extended to different baseline strategies.

        DPOV: Dynamic Participation of Volume - guarantees completion of the order

        Args:
            current_volumes: Current volumes for each environment
            action_values: Action values for each environment
            mode: Mode to use for trade size calculation, currently only 'DPOV' is supported
        Returns:
            Trade sizes for each environment
        """
        # set up paths for directing computation
        has_shares_remaining = self.shares_remaining > 0
        is_last_bin = self.current_step == self.time_horizon - 1
        non_last_bin = self.current_step < self.time_horizon - 1

        # otherwise we compute the expected pov to finish and adjust for the action
        # 1. Adjust pov based on realized trades and expected horizon volume
        # Each bin, adjust EHV based on shares_remaining/EHV[t,H]
        rem_pct_of_day = (self.time_horizon - self.current_step).float() / MINUTES_PER_DAY
        # ehv_adjusted = shares_remaining/(adv * rem_pct_of_day)
        ehv_remaining = torch.clamp(self.adv * rem_pct_of_day, min=1.0)
        pov_adjusted = self.shares_remaining.float() / ehv_remaining
        # 2. Adjust pov based on action
        # Constrain POV between 0 and POV_CAP
        action_pov = torch.clamp(pov_adjusted * (1.0 + action_values), min=0.0, max=POV_CAP)
        zeros = torch.zeros_like(current_volumes, dtype=torch.float32)

        # Proposed trade sizes for non-last bins
        proposed = torch.clamp(action_pov * current_volumes, min=0.0)
        proposed = torch.min(proposed, self.shares_remaining.float())

        # Execute trades: all remaining on last bin, else proposed; return integer shares
        trade_sizes_float = torch.where(
            has_shares_remaining & is_last_bin,
            self.shares_remaining.float(),
            torch.where(has_shares_remaining & non_last_bin, proposed, zeros),
        )
        trade_sizes = torch.round(trade_sizes_float).to(torch.int64)

        # Compute minute POV used for impact calculation as executed shares over minute volume
        achieved_minute_pov = torch.where(
            current_volumes > 0,
            trade_sizes.float() / current_volumes.clamp_min(1.0),
            zeros,
        )
        povs_per_minute = torch.where(
            has_shares_remaining,
            achieved_minute_pov,
            zeros,
        )
        
        return trade_sizes, povs_per_minute, action_pov
        
    
    def _calculate_arrival_cost(self, trade_sizes, fill_prices):
        """
        Calculates the trade cost for each environment.
        This is the slippage cost of the trade, where slippage is defined as the difference between the 
        fill price and the arrival price divided by the arrival price. The weight is the trade size divided 
        by the order quantity. The cost is the side * weight * price_performance. Thus a negative cost 
        is a profit.

        Args:
            trade_sizes: Trade sizes for each environment
            fill_prices: Fill prices for each environment

        Returns:
            Trade cost for each environment
        """
        # Sanity checks on order quantities and arrival prices
        if torch.any(self.order_qty <= 0):
            sample_idx = torch.nonzero(self.order_qty <= 0, as_tuple=False).view(-1)[:5].tolist()
            logger.warning(f"order_qty had {int((self.order_qty <= 0).sum().item())} non-positive entries; divisions may be unstable. sample idx={sample_idx}")
        
        if torch.any(~torch.isfinite(self.arrival_price)) or torch.any(self.arrival_price <= 0):
            bad_arrival_mask = (~torch.isfinite(self.arrival_price)) | (self.arrival_price <= 0)
            sample_idx = torch.nonzero(bad_arrival_mask, as_tuple=False).view(-1)[:5].tolist()
            if 'arrival_error' not in self.logged_errors:
                logger.warning(f"arrival_price had {int(bad_arrival_mask.sum().item())} invalid values; divisions may produce inf. sample idx={sample_idx}")
                self.logged_errors['arrival_error'] = True
        
        weight = trade_sizes / self.order_qty.float()
        price_performance = (fill_prices - self.arrival_price) / self.arrival_price

        if torch.any(~torch.isfinite(price_performance)):
            sample_idx = torch.nonzero(~torch.isfinite(price_performance), as_tuple=False).view(-1)[:5].tolist()
            if 'price_performance_error' not in self.logged_errors:
                logger.warning(f"price_performance produced non-finite values for {int((~torch.isfinite(price_performance)).sum().item())} envs; check arrival_price. sample idx={sample_idx}")
                self.logged_errors['price_performance_error'] = True
        
        trade_cost = self.arrival_cost_weight * self.side * price_performance * weight

        return trade_cost
        
    def _calculate_vwap_cost(self, trade_sizes, fill_prices, current_market_volume, current_market_vwap):
        """
        Calculate the slippage of market VWAP to order VWAP per environment.

        Outperformance yields negative cost; underperformance positive. Uses prior-step
        market data (the bar in which the trade executed) and weights by the fraction
        of the order executed in this step.
        """

        # Update cumulative market VWAP components
        self.cum_market_volume += current_market_volume
        self.cum_market_dollars += current_market_volume * current_market_vwap
        safe_cum_vol = torch.clamp(self.cum_market_volume, min=1.0)
        market_ivwap = self.cum_market_dollars / safe_cum_vol

        # Order VWAP to-date; zero before first fill (cost will be zero if no trade this step)
        order_vwap = self.order_vwap

        # Step weight: fraction of order executed this step
        weight = trade_sizes.float() / self.order_qty.clamp_min(1).float()

        # calculare vwap slippage, signed by side so outperformance is negative
        vwap_slippage = (order_vwap - market_ivwap) / market_ivwap
        vwap_cost = self.vwap_cost_weight * self.side * vwap_slippage * weight

        # Zero cost when no trade executed this step
        vwap_cost = torch.where(trade_sizes > 0, vwap_cost, torch.zeros_like(vwap_cost))

        return vwap_cost
        
    def _calculate_rate_penalty(self, current_volumes, sigma_step):
        """
        Calculates the rate penalty for each environment.

        Args:
            current_volumes: Current volumes for each environment
            sigma_step: Step size for the volatility

        Returns:
            Rate penalty for each environment
        """
        # sanity check on time horizon
        if torch.any(self.time_horizon <= 0):
            sample_idx = torch.nonzero(self.time_horizon <= 0, as_tuple=False).view(-1)[:5].tolist()
            if 'time_horizon_error' not in self.logged_errors:
                logger.warning(f"time_horizon had {int((self.time_horizon <= 0).sum().item())} non-positive entries; clipping to 1. sample idx={sample_idx}")
                self.logged_errors['time_horizon_error'] = True
        time_horizon_safe = self.time_horizon.clamp_min(1).float()
        
        # target completion ratio = 
        # traded_volume[order start, now]/(traded_volume[order start, now]+expected_volume[now+1,H])
        volume_traded = current_volumes[self.start_time:self.current_step].sum()
        # TODO: this needs to be modified if a volume profile exists
        expected_volume = self.adv * (time_horizon_safe - self.current_step).float()/MINUTES_PER_DAY
        target_completion_ratio = volume_traded / torch.clamp(volume_traded + expected_volume, min=1.0)
        target_completion_ratio = torch.clamp(target_completion_ratio, min=0.0, max=1.0)
        
        actual_completion_ratio = (self.order_qty - self.shares_remaining).float() / self.order_qty.float()
        rate_deviation = actual_completion_ratio - target_completion_ratio
        # Penalize deviation using a minutely volatility scale
        rate_penalty = sigma_step * torch.abs(rate_deviation)

        return rate_penalty
    
    def _calculate_unfilled_cost(self, trade_sizes, daily_sigma):
        """
        Calculates the unfilled cost for each environment.

        Args:
            trade_sizes: Trade sizes for each environment
            daily_sigma: Daily volatility for each environment

        Returns:
            Unfilled cost for each environment
        """
        unfilled_cost = torch.zeros_like(trade_cost)
        # use daily volatility as a proxy for overnight holding cost
        # TODO: ideally this is close to open vol (or a true trading cost)
        if truncated_mask.any():
            unfilled_ratio = self.shares_remaining[truncated_mask].float() / self.order_qty[truncated_mask]
            unfilled_cost[truncated_mask] = daily_sigma[truncated_mask] * unfilled_ratio

        return self.unfilled_cost_weight * unfilled_cost 

        
    def _calculate_holding_risk_cost(self, sigma_step):
        """
        Calculates the holding risk cost for each environment.

        Args:
            sigma_step: Step size for the volatility

        Returns:
            Holding risk cost for each environment
        """
        # holding risk penalty
        holding_risk_cost = self.risk_lambda * sigma_step * (self.shares_remaining.float() 
                    / self.order_qty.clamp_min(1).float()).abs()

        return self.holding_risk_weight * holding_risk_cost


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
        # Clamp daily_sigma to be non-negative and log anomalies
        raw_daily_sigma = market_data['daily_volatility_lag1']
        bad_sigma_mask = (~torch.isfinite(raw_daily_sigma)) | (raw_daily_sigma < 0)
        if torch.any(bad_sigma_mask):
            sample_idx = torch.nonzero(bad_sigma_mask, as_tuple=False).view(-1)[:5].tolist()
            logger.warning(f"daily_volatility_lag1 had {int(bad_sigma_mask.sum().item())} invalid values (neg or non-finite); sample idx={sample_idx}")
        daily_sigma = torch.clamp(raw_daily_sigma, min=0.0001)
        sigma_step = daily_sigma / torch.sqrt(torch.tensor(MINUTES_PER_DAY, device=daily_sigma.device))

        # Compute trade sizes
        fractions = self.action_values[self.actions]

        # Each bin, adjust EHV based on shares_remaining/EHV[t,H]
        rem_pct_of_day = (self.time_horizon - self.current_step).float() / MINUTES_PER_DAY
        # ehv_adjusted = shares_remaining/(adv * rem_pct_of_day)
        ehv_pct_remaining = self.ehv_pct * rem_pct_of_day
        ehv_adjusted = self.shares_remaining.float() / ehv_pct_remaining
        
        # Log if (1 + fractions) would be negative
        raw_one_plus = 1.0 + fractions
        if torch.any(raw_one_plus < 0):
            sample_idx = torch.nonzero(raw_one_plus < 0, as_tuple=False).view(-1)[:5].tolist()
            logger.warning(f"Action adjustment pushed (1 + fraction) negative for {int((raw_one_plus < 0).sum().item())} envs; clipping to 0. sample idx={sample_idx}")
                
        trade_sizes, povs_per_minute, action_pov = self._calculate_trade_sizes(current_volumes, fractions, mode='DPOV')

        # Update market and order cumulative stats for VWAP and completion
        self.total_market_volume += current_volumes + trade_sizes.float()
        self.total_market_volume = torch.clamp(self.total_market_volume, min=1)
        # Cumulative market IVWAP
        self.cum_market_volume += current_volumes
        self.cum_market_dollars += current_volumes * vwap_prices

        # Compute impacts using environment-specific parameters
        immediate_impact_per_minute = self._calculate_deterministic_impact(povs_per_minute)
        # Assign immediate impact for this step
        self.immediate_impact = immediate_impact_per_minute

        # Residual impact decay using environment-specific parameters
        delta_t = torch.where(
            self.last_trade_step >= 0,
            (self.current_step - self.last_trade_step).float(),
            torch.ones_like(self.current_step, dtype=torch.float32)
        )
        # Avoid division by zero (or negative) in decay rate
        if torch.any(~torch.isfinite(self.env_decay_rate)) or torch.any(self.env_decay_rate <= 0):
            bad_tau_mask = (~torch.isfinite(self.env_decay_rate)) | (self.env_decay_rate <= 0)
            sample_idx = torch.nonzero(bad_tau_mask, as_tuple=False).view(-1)[:5].tolist()
            logger.warning(f"env_decay_rate (tau) had {int(bad_tau_mask.sum().item())} invalid values; clipping to min 0.5. sample idx={sample_idx}")
        decay_rate_safe = torch.clamp(self.env_decay_rate, min=0.5)
        lambda_decay = torch.exp(-delta_t / decay_rate_safe)
        self.accumulated_impact = lambda_decay * self.accumulated_impact + self.immediate_impact
        fill_prices = vwap_prices * (1 + self.accumulated_impact)
        # Diagnostics for non-finite fill prices
        if torch.any(~torch.isfinite(fill_prices)):
            bad_idx = torch.nonzero(~torch.isfinite(fill_prices), as_tuple=False).view(-1)[:5].tolist()
            logger.error(
                "Non-finite fill_prices detected; sample idx=%s | vwap=%s | accum_impact=%s | imm_impact=%s | pov_minute=%s",
                bad_idx,
                vwap_prices[bad_idx].detach().cpu().numpy(),
                self.accumulated_impact[bad_idx].detach().cpu().numpy(),
                self.immediate_impact[bad_idx].detach().cpu().numpy(),
                povs_per_minute[bad_idx].detach().cpu().numpy(),
            )

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

        # Compute costs either via pluggable cost model or legacy helpers
        if self.cost_model is not None:
            total_step_cost, components = self.cost_model.get_total_cost(
                side=self.side,
                order_qty=self.order_qty,
                arrival_price=self.arrival_price,
                fill_prices=fill_prices,
                trade_sizes=trade_sizes,
                order_vwap=self.order_vwap,
                sigma_step=sigma_step,
                shares_remaining=self.shares_remaining,
                adv=self.adv,
                time_horizon=self.time_horizon,
                current_step=self.current_step,
                cum_market_volume=self.cum_market_volume,
                cum_market_dollars=self.cum_market_dollars
            )
            arrival_cost = components['arrival_cost']
            vwap_cost = components['vwap_cost']
            rate_devitation_penalty = components['rate_penalty']
            hoding_cost = components['holding_risk_cost']
            unfilled_cost = components['unfilled_cost']
        else:
            arrival_cost = self._calculate_arrival_cost(trade_sizes, fill_prices)
            # Legacy rate penalty
            rate_devitation_penalty = self._calculate_rate_penalty(current_volumes, sigma_step)
            vwap_cost = self._calculate_vwap_cost(
                trade_sizes,
                fill_prices,
                current_market_volume=current_volumes,
                current_market_vwap=vwap_prices,
            )
            hoding_cost = self._calculate_holding_risk_cost(sigma_step)
            # Fallback unfilled cost (zeros for now)
            unfilled_cost = torch.zeros_like(arrival_cost)

            total_step_cost = arrival_cost + vwap_cost + rate_devitation_penalty + hoding_cost + unfilled_cost

        reward = -total_step_cost
        # Guard against non-finite rewards propagating into training, with logging
        if torch.any(~torch.isfinite(reward)):
            sample_idx = torch.nonzero(~torch.isfinite(reward), as_tuple=False).view(-1)[:5].tolist()
            if 'reward_error' not in self.logged_errors:
                logger.warning(f"Non-finite rewards detected for {int((~torch.isfinite(reward)).sum().item())} envs; converting to 0. sample idx={sample_idx}")
                self.logged_errors['reward_error'] = True
        reward = torch.nan_to_num(reward, nan=0.0, posinf=0.0, neginf=0.0)
        
        self.total_cost = getattr(self, "total_cost", 0.0) + total_step_cost.detach().sum()

        # Update done flags (keep episodes running until time horizon for plotting)
        self.done = truncated_mask

        # Get new observations
        obs = self._get_observation()

        if not np.isfinite(obs).all():
            idx = np.where(~np.isfinite(obs))[0]
            raise RuntimeError(f"step_asynch:Env returned non-finite features at idx={idx},values={obs[idx]}")
        
        # Convert to numpy for VecEnv interface
        # Note: This is not strictly necessary, but keeps interface consistent
        obs_np = obs
        rewards_np = reward.cpu().numpy()
        # Validate rewards are finite
        if not np.isfinite(rewards_np).all():
            raise RuntimeError(f"step_wait: non-finite rewards detected: {rewards_np[~np.isfinite(rewards_np)]}")
        dones_np = self.done.cpu().numpy()
        # Populate infos per environment with cost components for TensorBoard logging
        infos = []
        for i in range(self.num_envs):
            infos.append({
                'shares_remaining': self.shares_remaining[i].item(),
                'order_vwap': self.order_vwap[i].item(),
                'arrival_price': self.arrival_price[i].item(),
                'immediate_impact': self.immediate_impact[i].item(),
                'accumulated_impact': self.accumulated_impact[i].item(),
                'action_percentage': float(self.last_action_fraction[i].item()),
                'action_pov': float(action_pov[i].item()),
                'arrival_cost': float(arrival_cost[i].item()),
                'vwap_cost': float(vwap_cost[i].item()),
                'rate_penalty': float(rate_devitation_penalty[i].item()),
                'unfilled_cost': float(unfilled_cost[i].item()),
                'holding_risk_cost': float(hoding_cost[i].item()),
                'total_step_cost': float(total_step_cost[i].item()),
                'step_reward': float(reward[i].item()),
            })

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


    def generate_random_order_indices(self, num_episodes):
        """
        Generate random order indices for execution.
    
        Args:
            num_episodes: Number of random orders to select
    
        Returns:
            List of order indices
        """
        if not hasattr(self, '_np_random') or self._np_random is None:
            self._np_random = np.random.RandomState(self._seed)
    
        num_orders = len(self.orders_df)
        return self._np_random.choice(num_orders, size=num_episodes, replace=False).tolist()


    def execute_orders_vectorized(self, model, order_indices, collect_step_info=True):
        """
        Unified vectorized execution method that processes orders in parallel batches.
        Collects full step-by-step data for analytics while maintaining high performance.
    
        Args:
            model: The trained RL model
            order_indices: List of order indices to execute
            collect_step_info: If True, collect per-step info (default True for analytics)
    
        Returns:
            List of order execution data (list of lists of step dictionaries)
        """
        num_episodes = len(order_indices)
    
        # Process orders in batches for optimal performance
        batch_size = min(self.num_envs, num_episodes)
        num_batches = (num_episodes + batch_size - 1) // batch_size
    
        all_orders = []
    
        for batch_idx in range(num_batches):
            start_idx = batch_idx * batch_size
            end_idx = min(start_idx + batch_size, num_episodes)
            batch_indices = order_indices[start_idx:end_idx]
        
            # Execute this batch in parallel
            batch_orders = self._execute_batch_parallel(model, batch_indices, collect_step_info)
            all_orders.extend(batch_orders)
    
        return all_orders


    def _execute_batch_parallel(self, model, batch_indices, collect_step_info):
        """
        Execute a batch of orders in parallel using vectorized environments.
    
        Args:
            model: The trained model
            batch_indices: List of order indices for this batch
            collect_step_info: Whether to collect step-by-step information
    
        Returns:
            List of order execution data for this batch
        """
        batch_size = len(batch_indices)
    
        # Pad batch if smaller than num_envs (use dummy orders that we'll ignore)
        if batch_size < self.num_envs:
            padding_indices = [batch_indices[0]] * (self.num_envs - batch_size)
            full_indices = batch_indices + padding_indices
        else:
            full_indices = batch_indices
    
        # Reset all environments with the batch orders
        self.order_idx = torch.tensor(full_indices, device=self.device)
        obs = self.reset()
    
        if collect_step_info:
            # Pre-allocate tensors for efficiency
            max_horizon = self.time_horizon[:batch_size].max().item()
        
            # Store all step data in tensors for speed
            step_data = {
                'shares_remaining': torch.zeros((batch_size, max_horizon), device=self.device),
                'last_fill_price': torch.zeros((batch_size, max_horizon), device=self.device),
                'last_trade_size': torch.zeros((batch_size, max_horizon), device=self.device),
                'immediate_impact': torch.zeros((batch_size, max_horizon), device=self.device),
                'accumulated_impact': torch.zeros((batch_size, max_horizon), device=self.device),
                'order_vwap': torch.zeros((batch_size, max_horizon), device=self.device),
                'action_percentage': torch.zeros((batch_size, max_horizon), device=self.device),
                'mid_price': torch.zeros((batch_size, max_horizon), device=self.device),
                'vwap_price': torch.zeros((batch_size, max_horizon), device=self.device),
                'current_trade_volume': torch.zeros((batch_size, max_horizon), device=self.device),
            }
        
            # Cache constant order information
            order_constants = []
            for i in range(batch_size):
                order_constants.append({
                    'order_idx': self.order_idx[i].item(),
                    'ticker': self.tickers[i],
                    'order_qty': self.order_qty[i].item(),
                    'adv_pct': self.adv_pct[i].item(),
                    'ehv_pct': self.ehv_pct[i].item(),
                    'start_time': self.start_time[i].item(),
                    'end_time': self.end_time[i].item(),
                    'time_horizon': self.time_horizon[i].item(),
                    'side': 'buy' if self.side[i].item() == 1 else 'sell',
                    'arrival_price': self.arrival_price[i].item(),
                    'adv': self.adv[i].item(),
                    'date': str(self.order_dates[i]) if self.order_dates else None,
                })
        
        step_counts = torch.zeros(batch_size, dtype=torch.int32, device=self.device)
    
    # Track which environments are done
    done = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
    step = 0
    
    while not done[:batch_size].all():
        # Get actions for all active environments
        actions = np.zeros(self.num_envs, dtype=np.int32)
        for i in range(self.num_envs):
            if not done[i]:
                if hasattr(model, 'predict'):
                    action, _ = model.predict(obs[i], deterministic=False)
                else:
                    action, _ = model.predict(obs[i])
                actions[i] = action
        
        # Step all environments
        self.step_async(actions)
        obs, rewards, dones, infos = self.step_wait()
        
        if collect_step_info and step < max_horizon:
            # Get market data once for all environments
            market_data = self._get_market_data_batch(
                ['trade_volume', 'vwap', 'trade_high', 'trade_low'],
                use_prior_step=True
            )
            
            # Store data for active environments in this batch
            active_mask = ~done[:batch_size] 
            for i in range(batch_size):
                if active_mask[i]:
                    step_data['shares_remaining'][i, step] = self.shares_remaining[i]
                    step_data['last_fill_price'][i, step] = self.last_fill_price[i]
                    step_data['last_trade_size'][i, step] = self.last_trade_size[i]
                    step_data['immediate_impact'][i, step] = self.immediate_impact[i]
                    step_data['accumulated_impact'][i, step] = self.accumulated_impact[i]
                    step_data['order_vwap'][i, step] = self.order_vwap[i]
                    step_data['action_percentage'][i, step] = self.last_action_fraction[i]
                    step_data['mid_price'][i, step] = (market_data['trade_high'][i] + market_data['trade_low'][i]) * 0.5
                    step_data['vwap_price'][i, step] = market_data['vwap'][i]
                    step_data['current_trade_volume'][i, step] = market_data['trade_volume'][i]
                    step_counts[i] = step + 1
        
        done = torch.tensor(dones, device=self.device, dtype=torch.bool)
        step += 1
    
    # Convert tensor data to list format for compatibility
    batch_orders = []
    
    if collect_step_info:
        for i in range(batch_size):
            order_info = []
            num_steps = step_counts[i].item()
            
            for s in range(num_steps):
                info = order_constants[i].copy()
                info.update({
                    'current_step': s,
                    'episode': batch_indices[i],
                    'shares_remaining': step_data['shares_remaining'][i, s].item(),
                    'last_fill_price': step_data['last_fill_price'][i, s].item(),
                    'last_trade_size': step_data['last_trade_size'][i, s].item(),
                    'immediate_impact': step_data['immediate_impact'][i, s].item(),
                    'accumulated_impact': step_data['accumulated_impact'][i, s].item(),
                    'order_vwap': step_data['order_vwap'][i, s].item(),
                    'action_percentage': step_data['action_percentage'][i, s].item(),
                    'mid_price': step_data['mid_price'][i, s].item(),
                    'vwap_price': step_data['vwap_price'][i, s].item(),
                    'current_trade_volume': step_data['current_trade_volume'][i, s].item(),
                    'total_cost': self.total_cost[i].item(),
                })
                order_info.append(info)
            
            batch_orders.append(order_info)
    else:
        # Just return empty lists if not collecting step info
        batch_orders = [[] for _ in range(batch_size)]
    
    return batch_orders



    def execute_orders(self, model, num_episodes=10, fixed_order_indices=None, collect_step_info=False):
        """
        Execute orders using the provided model (maintains same interface as original).
        @param model: The trained RL model to use for action selection.
        @param num_episodes: Number of orders to execute in this run.
        @param fixed_order_indices: Optional list of specific order indices to use for consistent comparison.
                                  If None, random orders will be selected. If provided, num_episodes will be
                                  set to len(fixed_order_indices).
        @param collect_step_info: If True, collect per-step info dicts (slower). If False, only per-episode.
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
                
                if collect_step_info:
                    # Get market data for the bin in which the trade executed (prior step after step_wait)
                    market_data = self._get_market_data_batch(['trade_volume', 'vwap', 'trade_high', 'trade_low'], use_prior_step=True)
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
                        'total_cost': self.total_cost[env_idx].item(),
                        'action_percentage': float(self.last_action_fraction[env_idx].item()),
                        'adv': self.adv[env_idx].item(),
                        'current_step': step,
                        'episode': ep,
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
        """
        Reset environment with a specific order index.
        Args:
            order_idx: The index of the order to reset the environment to.
        Returns:
            The initial observation of the environment.
        """
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
                    if 'Y_error' in self.logged_errors:
                        logger.warning(f"Found {nan_mask.sum()} NaN Y values, replacing with median {median_value}")
                        self.logged_errors['Y_error'] = True
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
        self.cum_market_volume = torch.zeros(self.num_envs, device=self.device, dtype=torch.float32)
        self.cum_market_dollars = torch.zeros(self.num_envs, device=self.device, dtype=torch.float32)
        
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