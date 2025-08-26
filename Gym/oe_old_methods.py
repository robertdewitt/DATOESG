
"""
This file contains the old methods for the optimal execution environment.

"""


def _calculate_unfilled_cost(self, trade_sizes, daily_sigma, truncated_mask,trade_cost):
        """
        Calculates the unfilled cost for each environment.

        Args:
            trade_sizes: Trade sizes for each environment
            daily_sigma: Daily volatility for each environment

        Returns:
            Unfilled cost for each environment
        """
        unfilled_cost = torch.zeros_like(trade_sizes)
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

# in step_wait if cost_model is not defined
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
            holding_cost = self._calculate_holding_risk_cost(sigma_step)
            # Fallback unfilled cost (zeros for now)
            unfilled_cost = torch.zeros_like(arrival_cost)

            total_step_cost = arrival_cost + vwap_cost + rate_devitation_penalty + holding_cost + unfilled_cost

def execute_orders_old(self, model, num_episodes=10, fixed_order_indices=None, collect_step_info=False):
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