# class which create orders for the gym
import pandas as pd
import numpy as np
import random
import mkt_data_yfinance as mdy
import logging
from typing import Optional, List, Dict
import time
from quote_and_trade_analytics import QuoteAndTradeAnalytics
from datetime import date
from propagator_param_loader import PropagatorParamLoader

# Set up logger for this module
logger = logging.getLogger(__name__)


class OrderGenerator:
    def __init__(self, stock_df_list, market_data, num_orders=10,  min_adv_pct=0.0001,
                 max_adv_pct=0.25, min_time_horizon=2, max_time_horizon=390, sd_delta=0, 
                 ed_delta=0, md_source='yfinance', debug=False, seed=None, dates=None,
                 analytics: Optional['QuoteAndTradeAnalytics'] = None,
                 propagator_loader: Optional['PropagatorParamLoader'] = None,
                 y_column: str = 'Y', tau_column: str = 'tau'):
        """
        Initialize the OrderGenerator with a random DataFrame of orders.
        Args:
            stock_df_list: List of DataFrames containing stock data.
            market_data: Instance of MarketDataLoader to fetch market data.
            num_orders: Number of random orders to generate.
            min_adv_pct: Minimum percentage of ADV for orders.
            max_adv_pct: Maximum percentage of ADV for orders.
            min_time_horizon: Minimum time horizon for orders in minutes.   
            max_time_horizon: Maximum time horizon for orders in minutes.
            sd_delta: Start date delta - number of days to shift the start date of the order generation.
            ed_delta: End date delta - number of days to shift the end date of the order generation.
            md_source: Source of market data - 'yfinance' or 'mana'.
            debug: Enable debug logging.
            seed: Random seed for reproducibility.
            analytics: QuoteAndTradeAnalytics instance with pre-loaded analytics data.
            propagator_loader: PropagatorParamLoader instance for Y and tau parameters (optional).
            y_column: Column name for Y parameter in propagator data (default: 'Y').
            tau_column: Column name for tau parameter in propagator data (default: 'tau').
            dates: List of dates to generate orders for.
        Returns:
            None
        """
        # Set logging level based on debug parameter
        if debug:
            logger.setLevel(logging.DEBUG)

        # Set seed BEFORE any random operations
        self.seed = seed
        if self.seed is not None:
            np.random.seed(self.seed)
            random.seed(self.seed)
            logger.debug(f"Random seed set to {self.seed} for both numpy and random modules")

        self.stock_df_list = stock_df_list
        self.num_orders = num_orders
        self.min_adv_pct = min_adv_pct
        self.max_adv_pct = max_adv_pct
        self.min_time_horizon = min_time_horizon
        self.max_time_horizon = max_time_horizon
        self.mdl = market_data
        self.sd_delta = sd_delta
        self.ed_delta = ed_delta
        self.md_source = md_source
        self.analytics = analytics  # Store analytics instance
        self.propagator_loader = propagator_loader  # Store propagator loader instance
        self.y_column = y_column
        self.tau_column = tau_column
        self.dates = dates

        if self.dates is None:
            logger.warning("No dates provided, cannot generate orders")
            return

        self.orders_df = self._generate_orders_fast()


    def _generate_orders_fast(self):
        """
        Generate a dataframe of random orders based on the stock data.
        Optimized version using vectorized operations throughout.
        Returns:
            orders_df: DataFrame containing the generated orders.
        """
        import time
        start_time = time.time()
        
        # Set up random number generator
        rng = np.random.RandomState(self.seed)
        
        # Extract unique stock names and dates
        available_stocks = list(self.stock_df_list.keys())
        unique_dates = self.dates
        
        if len(unique_dates) == 0:
            raise ValueError(f"No dates available")
        
        logger.info(f"Generating {self.num_orders} orders for {len(available_stocks)} stocks across {len(unique_dates)} dates")
        logger.info(f"Date range: {unique_dates[0]} to {unique_dates[-1]}")
        
        # Step 1: Build availability matrix and ADV matrix
        logger.info("Building availability matrix and computing ADV...")
        matrix_start = time.time()
        
        availability_matrix = self._build_availability_matrix(available_stocks, unique_dates)
        adv_matrix = self.analytics.get_adv_matrix(available_stocks, unique_dates)
        
        logger.info(f"Matrix built in {time.time() - matrix_start:.2f}s")
        
        # Step 2: Pre-filter population by analytics and propagator quality BEFORE sampling
        logger.info("Filtering population by volatility and R^2 before sampling...")
        filter_start = time.time()
        base_valid_mask = (availability_matrix > 0) & (adv_matrix > 0)
        valid_indices = np.argwhere(base_valid_mask)
        if valid_indices.size == 0:
            raise ValueError("No valid stock-date pairs after base availability/ADV filter")

        # Build pairs (symbol, date) for base-valid slots
        symbol_date_pairs = [(available_stocks[i], unique_dates[j]) for i, j in valid_indices]
        # Load daily volatility for these pairs
        analytics_dict = self.analytics.get_analytics_bulk(symbol_date_pairs) if self.analytics is not None else {}
        # Load propagator params including R^2 for these pairs
        params_dict = {}
        if self.propagator_loader is not None:
            try:
                params_dict = self.propagator_loader.get_params_batch(
                    symbol_date_pairs,
                    fallback_days=5,
                    y_column=self.y_column,
                    tau_column=self.tau_column,
                    include_r2=True
                )
            except Exception as e:
                logger.warning(f"Propagator get_params_batch failed for prefilter: {e}")

        # Build allowed mask of same shape as matrices
        allowed_pairs_mask = np.zeros_like(base_valid_mask, dtype=bool)
        # Prepare date normalization for propagator keys
        from datetime import datetime as _dt
        date_fmt = getattr(self.propagator_loader, 'date_format', '%Y%m%d') if self.propagator_loader is not None else '%Y%m%d'

        for (i, j) in valid_indices:
            sym = available_stocks[i]
            dt = unique_dates[j]
            key = (sym, dt)
            # analytics dv
            dv_ok = False
            if key in analytics_dict:
                dv = analytics_dict[key].get('daily_volatility', None)
                try:
                    dv_val = float(dv)
                    dv_ok = np.isfinite(dv_val) and (dv_val > 0.0) and (dv_val <= 0.5)
                except Exception:
                    dv_ok = False
            # r2
            r2_ok = True  # default to True if missing
            if params_dict:
                # Match propagator dict keys: (SYMBOL_UPPER, normalized_date_str)
                if isinstance(dt, str):
                    norm_date = dt
                elif isinstance(dt, _dt) or hasattr(dt, 'strftime'):
                    try:
                        norm_date = dt.strftime(date_fmt)
                    except Exception:
                        norm_date = str(dt)
                else:
                    norm_date = str(dt)
                pkey = (sym.upper(), norm_date)
                if pkey in params_dict and isinstance(params_dict[pkey], dict):
                    r2 = params_dict[pkey].get('r2', None)
                    try:
                        r2_ok = (r2 is not None) and np.isfinite(float(r2)) and (float(r2) >= 0.02)
                    except Exception:
                        r2_ok = False
            if dv_ok and r2_ok:
                allowed_pairs_mask[i, j] = True

        if not allowed_pairs_mask.any():
            raise ValueError("No stock-date pairs passed volatility and R^2 filters before sampling")

        logger.info(f"Population filtered in {time.time() - filter_start:.2f}s; allowed pairs: {int(allowed_pairs_mask.sum())}")

        # Step 3: Generate valid orders (from filtered population only)
        number_of_valid_stocks = int(allowed_pairs_mask.sum())
        logger.info(f"Generating valid orders from filtered population...")

        gen_start = time.time()
        
        valid_orders_data = self._sample_valid_orders(
            availability_matrix, adv_matrix, available_stocks, unique_dates, 
            self.num_orders, rng, allowed_pairs_mask=allowed_pairs_mask
        )
        
        logger.info(f"Orders sampled in {time.time() - gen_start:.2f}s")
        
        # Step 3: Generate order parameters
        logger.info("Generating order parameters...")
        param_start = time.time()
        
        order_params = self._generate_order_parameters(
            valid_orders_data['minutes'],
            valid_orders_data['advs'],
            len(valid_orders_data['stocks']),
            rng
        )
        
        logger.info(f"Parameters generated in {time.time() - param_start:.2f}s")
        
        # Step 4: Create DataFrame
        orders_df = self._create_orders_dataframe(valid_orders_data, order_params)
        
        # Step 5: Load analytics
        logger.info("Loading analytics...")
        analytics_start = time.time()
        
        orders_df = self._load_analytics_data(orders_df)
        
        logger.info(f"Analytics loaded in {time.time() - analytics_start:.2f}s")
        
        # Step 6: Load propagator parameters
        logger.info("Loading propagator parameters...")
        prop_start = time.time()
        
        orders_df = self._load_propagator_params(orders_df)
        
        logger.info(f"Propagator parameters loaded in {time.time() - prop_start:.2f}s")
        
        # Step 6.1: Post-sampling quality filter (diagnostic) — confirm prefilter effectiveness
        pre_filter_count = len(orders_df)
        orders_df = self._filter_orders_quality(orders_df)
        dropped = pre_filter_count - len(orders_df)
        if dropped > 0:
            logger.info(f"Quality filter dropped {dropped} orders (kept {len(orders_df)}).")
        
        # Step 7: Calculate intra-order returns
        logger.info("Calculating intra-order returns...")
        returns_start = time.time()
        
        orders_df = self._calculate_intra_order_returns(orders_df)
        
        logger.info(f"Returns calculated in {time.time() - returns_start:.2f}s")
        
        total_time = time.time() - start_time
        logger.info(f"Generated {len(orders_df)} orders in {total_time:.2f}s total")
        logger.debug(f"Orders per date distribution:\n{orders_df['date'].value_counts().sort_index()}")
        
        return orders_df

    def _filter_orders_quality(self, orders_df: pd.DataFrame) -> pd.DataFrame:
        """
        Drop orders that fail basic quality thresholds:
          - Missing or invalid daily_volatility (<=0) or daily_volatility > 0.5
          - Propagator R^2 < 0.02 when available
        Logs each dropped order with details.
        """
        if orders_df.empty:
            return orders_df
        df = orders_df.copy()
        # Use daily_volatility; fallback to daily_volatility_lag1 only for logging if dv missing
        dv = pd.to_numeric(df.get('daily_volatility', np.nan), errors='coerce')
        bad_vol_mask = (~np.isfinite(dv)) | (dv <= 0) | (dv > 0.5)

        # Flexible R^2 presence: look for 'r2' if populated by propagator loader
        r2_series = pd.to_numeric(df.get('r2', np.nan), errors='coerce') if ('r2' in df.columns) else np.nan
        has_r2 = isinstance(r2_series, pd.Series)
        low_r2_mask = (r2_series < 0.02) if has_r2 else pd.Series(False, index=df.index)

        drop_mask = bad_vol_mask | low_r2_mask
        if drop_mask.any():
            to_drop = df[drop_mask]
            for _, row in to_drop.iterrows():
                logger.warning(
                    "Dropping generated order: ticker=%s date=%s reason=%s dv=%s r2=%s",
                    row.get('ticker', 'NA'),
                    row.get('date', 'NA'),
                    ("bad_vol" if ((not np.isfinite(row.get('daily_volatility', np.nan))) or (row.get('daily_volatility', 0) <= 0) or (row.get('daily_volatility', 1) > 0.5)) else "low_r2"),
                    row.get('daily_volatility', None),
                    row.get('r2', None)
                )
            df = df[~drop_mask].reset_index(drop=True)
        return df


    def _build_availability_matrix(self, stocks: List[str], dates: List[date]) -> np.ndarray:
        """
        Build matrix of available minutes for each stock-date combination.
        @param stocks: List of stock symbols
        @param dates: List of dates
        @return: 2D numpy array (stocks x dates) with minute counts
        """
        availability_matrix = np.zeros((len(stocks), len(dates)), dtype=int)

        print(f"Availability Matrix Stocks: {stocks[0]} to {stocks[-1]}")
        print(f"Availability Matrix Dates: {dates[0]} to {dates[-1]}")
        
        for i, stock in enumerate(stocks):
            df = self.stock_df_list[stock]
            
            for j, date in enumerate(dates):
                if 'date' in df.columns:
                    minutes = len(df[df['date'] == date])
                elif hasattr(df.index, 'date'):
                    minutes = (df.index.date == date).sum()
                else:
                    minutes = (df.index == date).sum()
                
                availability_matrix[i, j] = minutes
        
        return availability_matrix

    def _sample_valid_orders(self, availability_matrix: np.ndarray, adv_matrix: np.ndarray,
                           stocks: List[str], dates: List[date], num_orders: int, 
                           rng: np.random.RandomState, allowed_pairs_mask: Optional[np.ndarray] = None) -> Dict:
        """
        Sample valid stock-date pairs for order generation.
        @param availability_matrix: Matrix of available minutes
        @param adv_matrix: Matrix of ADV values
        @param stocks: List of stock symbols
        @param dates: List of dates
        @param num_orders: Number of orders to generate
        @param rng: Random number generator
        @return: Dictionary with order data
        """
        # Find valid stock-date pairs (availability & ADV), intersect with allowed_pairs_mask if provided
        base_mask = (availability_matrix > 0) & (adv_matrix > 0)
        if allowed_pairs_mask is not None:
            if allowed_pairs_mask.shape != base_mask.shape:
                raise ValueError("allowed_pairs_mask shape must match availability/adv matrices")
            base_mask = base_mask & allowed_pairs_mask
        valid_pairs = np.argwhere(base_mask)
        
        if len(valid_pairs) == 0:
            # Debug information to help diagnose the issue
            availability_count = np.sum(availability_matrix > 0)
            adv_count = np.sum(adv_matrix > 0)
            total_pairs = availability_matrix.size
            
            logger.error(f"No valid stock-date pairs found:")
            logger.error(f"  Availability matrix: {availability_count}/{total_pairs} non-zero entries")
            logger.error(f"  ADV matrix: {adv_count}/{total_pairs} non-zero entries")
            logger.error(f"  Stocks: {len(stocks)} - {stocks[:5]}{'...' if len(stocks) > 5 else ''}")
            logger.error(f"  Dates: {len(dates)} - {dates[:5]}{'...' if len(dates) > 5 else ''}")
            
            # Show some sample values from each matrix
            if availability_matrix.size > 0:
                logger.error(f"  Availability matrix sample: {availability_matrix[:min(3, availability_matrix.shape[0]), :min(3, availability_matrix.shape[1])]}")
            if adv_matrix.size > 0:
                logger.error(f"  ADV matrix sample: {adv_matrix[:min(3, adv_matrix.shape[0]), :min(3, adv_matrix.shape[1])]}")
            
            raise ValueError("No valid stock-date pairs found with both data and ADV")
        
        # Oversample by 20% to ensure we get enough valid orders
        num_to_generate = int(num_orders * 1.2)
        sampled_indices = rng.choice(len(valid_pairs), size=num_to_generate, replace=True)
        
        # Extract stock and date indices
        stock_indices = valid_pairs[sampled_indices, 0]
        date_indices = valid_pairs[sampled_indices, 1]
        
        # Get actual values
        order_stocks = [stocks[i] for i in stock_indices]
        order_dates = [dates[i] for i in date_indices]
        order_minutes = availability_matrix[stock_indices, date_indices]
        order_advs = adv_matrix[stock_indices, date_indices]
        
        # Filter and truncate
        valid_mask = (order_minutes > 0) & (order_advs > 0)
        order_stocks = [s for s, v in zip(order_stocks, valid_mask) if v][:num_orders]
        order_dates = [d for d, v in zip(order_dates, valid_mask) if v][:num_orders]
        order_minutes = order_minutes[valid_mask][:num_orders]
        order_advs = order_advs[valid_mask][:num_orders]
        
        return {
            'stocks': order_stocks,
            'dates': order_dates,
            'minutes': order_minutes,
            'advs': order_advs
        }

    def _generate_order_parameters(self, order_minutes: np.ndarray, order_advs: np.ndarray, 
                                 num_orders: int, rng: np.random.RandomState) -> Dict:
        """
        Generate random parameters for orders (time horizons, quantities, etc).
        @param order_minutes: Available minutes for each order
        @param order_advs: ADV values for each order
        @param num_orders: Number of orders
        @param rng: Random number generator
        @return: Dictionary with order parameters
        """
        # Generate time horizons
        max_horizons = np.minimum(self.max_time_horizon, order_minutes)
        time_horizons = rng.randint(self.min_time_horizon, max_horizons + 1)
        
        # Generate start times
        max_starts = np.maximum(0, order_minutes - time_horizons)
        uniform_randoms = rng.random(num_orders)
        start_times = (uniform_randoms * (max_starts + 1)).astype(int)
        end_times = start_times + time_horizons
        
        # Calculate order sizes
        pct_of_day = np.maximum(time_horizons / order_minutes, 0.001)
        min_ehv_pct = self.min_adv_pct * pct_of_day
        max_ehv_pct = self.max_adv_pct * pct_of_day
        
        adv_pct = rng.uniform(min_ehv_pct, max_ehv_pct, size=num_orders)
        order_quantities = np.round(order_advs * adv_pct).astype(int)
        sides = rng.choice(['buy', 'sell'], size=num_orders)
        
        return {
            'time_horizons': time_horizons,
            'start_times': start_times,
            'end_times': end_times,
            'adv_pct': adv_pct,
            'ehv_pct': adv_pct / pct_of_day,
            'order_quantities': order_quantities,
            'sides': sides
        }

    def _create_orders_dataframe(self, valid_orders_data: Dict, order_params: Dict) -> pd.DataFrame:
        """
        Create the orders DataFrame from order data and parameters.
        @param valid_orders_data: Dictionary with stocks, dates, minutes, advs
        @param order_params: Dictionary with order parameters
        @return: Orders DataFrame
        """
        orders_df = pd.DataFrame({
            'ticker': valid_orders_data['stocks'],
            'order_qty': order_params['order_quantities'],
            'adv_pct': order_params['adv_pct'],
            'ehv_pct': order_params['ehv_pct'],
            'adv': valid_orders_data['advs'],
            'date': valid_orders_data['dates'],
            'start_time': order_params['start_times'],
            'end_time': order_params['end_times'],
            'time_horizon': order_params['time_horizons'],
            'side': order_params['sides'],
            # Initialize all analytics columns
            'avg_spread_21_days': 0.0,
            'avg_depth_21_days': 0.0,
            'daily_volatility': 0.0,
            'daily_volatility_lag1': 0.0,
            'daily_volatility_5d': 0.0,
            'daily_vwap': 0.0,
            'intra_order_return': 0.0,
            'adv_21_days': 0.0,
            'avg_trade_count_21_days': 0.0
        })
        
        return orders_df

    def _load_analytics_data(self, orders_df: pd.DataFrame) -> pd.DataFrame:
        """
        Load analytics data for all orders.
        @param orders_df: Orders DataFrame
        @return: Orders DataFrame with analytics data
        """
        # Get unique ticker-date pairs
        unique_pairs = orders_df[['ticker', 'date']].drop_duplicates()
        symbol_date_pairs = list(unique_pairs.itertuples(index=False, name=None))
        
        # Bulk load all analytics at once
        analytics_dict = self.analytics.get_analytics_bulk(symbol_date_pairs)
        
        if analytics_dict:
            # Create a mapping for fast lookup
            def get_analytics_value(row, field):
                key = (row['ticker'], row['date'])
                if key in analytics_dict:
                    return analytics_dict[key].get(field, 0.0)
                return 0.0
            
            # Apply to all analytics columns
            analytics_cols = {
                'adv_21_days': 'adv_21_days',
                'avg_spread_21_days': 'avg_spread_21_days',
                'avg_trade_count_21_days': 'avg_trade_count_21_days',
                'avg_depth_21_days': 'avg_depth_21_days',
                'daily_volatility': 'daily_volatility',
                'daily_volatility_lag1': 'daily_volatility_lag1',
                'daily_volatility_5d': 'daily_volatility_5d',
                'daily_vwap': 'vwap'
            }
            
            for df_col, analytics_col in analytics_cols.items():
                orders_df[df_col] = orders_df.apply(
                    lambda row: get_analytics_value(row, analytics_col), axis=1
                )
        
        return orders_df

    def _load_propagator_params(self, orders_df: pd.DataFrame) -> pd.DataFrame:
        """
        Load propagator parameters (Y and tau) for all orders using vectorized operations.

        Args:
            orders_df: Orders DataFrame
            fallback_days: Number of days to use for fallback if no propagator parameters are found

        Returns:    
            Orders DataFrame with Y and tau columns added

        """
        if self.propagator_loader is None:
            logger.info("No propagator loader provided, skipping Y and tau parameter loading")
            return orders_df
            
        logger.info("Loading propagator parameters (Y and tau)...")
        
        # Get unique ticker-date pairs for batch loading
        unique_pairs = orders_df[['ticker', 'date']].drop_duplicates()
        
        if unique_pairs.empty:
            logger.warning("No unique ticker-date pairs found")
            orders_df['Y'] = np.nan
            orders_df['tau'] = np.nan
            return orders_df
        
        # Convert to list of (symbol, date) tuples for batch loading
        symbol_date_pairs = [(row['ticker'], row['date']) for _, row in unique_pairs.iterrows()]
        
        # Batch load all parameters at once
        try:
            params_dict = self.propagator_loader.get_params_batch(
                symbol_date_pairs,
                fallback_days=5,
                y_column=self.y_column,
                tau_column=self.tau_column,
                include_r2=True
            )

            # Convert to DataFrame for efficient merging
            params_data = []
            for (symbol, date), vals in params_dict.items():
                Y = np.nan
                tau = np.nan
                r2 = np.nan
                try:
                    if isinstance(vals, dict):
                        Y = vals.get(self.y_column, vals.get('Y', np.nan))
                        tau = vals.get(self.tau_column, vals.get('tau', np.nan))
                        r2 = vals.get('r2', np.nan)
                    else:
                        # tuple or list
                        if len(vals) >= 1:
                            Y = vals[0]
                        if len(vals) >= 2:
                            tau = vals[1]
                        if len(vals) >= 3:
                            r2 = vals[2]
                except Exception:
                    pass
                params_data.append({
                    'ticker': symbol,
                    'date': date,
                    'Y': Y,
                    'tau': tau,
                    'r2': r2
                })
            
            if params_data:
                params_df = pd.DataFrame(params_data)
                
                orders_df['date'] = pd.to_datetime(orders_df['date']).dt.date
                params_df['date'] = pd.to_datetime(params_df['date'], format='%Y%m%d', errors='coerce').dt.date

                # Vectorized merge - much faster than apply()
                orders_df = orders_df.merge(
                    params_df[['ticker', 'date', 'Y', 'tau', 'r2']], 
                    on=['ticker', 'date'], 
                    how='left'
                )

            else:
                logger.warning("No propagator parameters loaded")
                orders_df['Y'] = np.nan
                orders_df['tau'] = np.nan
                
        except Exception as e:
            logger.error(f"Failed to batch load propagator parameters: {e}")
            orders_df['Y'] = np.nan
            orders_df['tau'] = np.nan
            return orders_df
        
        # Log statistics
        Y_count = orders_df['Y'].notna().sum()
        tau_count = orders_df['tau'].notna().sum()
        total_orders = len(orders_df)
        
        logger.info(f"Loaded Y parameters for {Y_count}/{total_orders} orders ({Y_count/total_orders*100:.1f}%)")
        logger.info(f"Loaded tau parameters for {tau_count}/{total_orders} orders ({tau_count/total_orders*100:.1f}%)")
        
        if Y_count > 0:
            Y_stats = orders_df['Y'].describe()
            logger.info(f"Y statistics: mean={Y_stats['mean']:.6f}, std={Y_stats['std']:.6f}, range=[{Y_stats['min']:.6f}, {Y_stats['max']:.6f}]")
        
        if tau_count > 0:
            tau_stats = orders_df['tau'].describe()
            logger.info(f"tau statistics: mean={tau_stats['mean']:.2f}, std={tau_stats['std']:.2f}, range=[{tau_stats['min']:.2f}, {tau_stats['max']:.2f}]")
        
        return orders_df

    def _calculate_intra_order_returns(self, orders_df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate intra-order returns for all orders.
        Positive return reflects a favorable move for the order side
        (up for buys, down for sells).
        @param orders_df: Orders DataFrame
        @return: Orders DataFrame with intra_order_return calculated
        """
        # Group by ticker-date for efficient processing
        for (ticker, date), group_indices in orders_df.groupby(['ticker', 'date']).groups.items():
            df = self.stock_df_list[ticker]
            
            # Get data for this date
            if 'date' in df.columns:
                date_data = df[df['date'] == date]
            elif hasattr(df.index, 'date'):
                date_data = df[df.index.date == date]
            else:
                date_data = df[df.index == date]
            
            if date_data.empty or 'bid_price' not in date_data.columns or 'ask_price' not in date_data.columns:
                continue
            
            # Calculate returns for each order in this group
            for idx in group_indices:
                row = orders_df.loc[idx]
                start_idx = min(row['start_time'], len(date_data) - 1)
                end_idx = min(row['end_time'] - 1, len(date_data) - 1)
                
                if start_idx >= 0 and end_idx >= 0 and end_idx > start_idx:
                    try:
                        bid_start = date_data.iloc[start_idx]['bid_price']
                        ask_start = date_data.iloc[start_idx]['ask_price']
                        bid_end = date_data.iloc[end_idx]['bid_price']
                        ask_end = date_data.iloc[end_idx]['ask_price']
                        
                        if bid_start > 0 and ask_start > 0 and bid_end > 0 and ask_end > 0:
                            mid_start = (bid_start + ask_start) / 2
                            mid_end = (bid_end + ask_end) / 2
                            raw_return = (mid_end - mid_start) / mid_start
                            # Side-adjust so positive means favorable move
                            order_side = str(row.get('side', 'buy')).lower()
                            intra_return = raw_return if order_side == 'buy' else -raw_return
                            orders_df.loc[idx, 'intra_order_return'] = intra_return
                    except Exception as e:
                        logger.debug(f"Error calculating return for {ticker} order {idx}: {e}")
        
        return orders_df

    @staticmethod
    def get_dates_from_dataframe(df):
        """
        Extract dates from a DataFrame, handling both date-as-column and date-as-index cases.
        @param df: DataFrame to extract dates from
        @return: Array of unique dates
        """
        if 'date' in df.columns:
            # Date is a column
            return df['date'].unique()
        elif hasattr(df.index, 'date'):
            # Date is index (DatetimeIndex) - extract date part
            return pd.Series(df.index.date).unique()
        elif df.index.name == 'date':
            # Index is named 'date' and contains date objects directly
            return pd.Series(df.index).unique()
        else:
            # Try to find any date-like columns
            date_cols = [col for col in df.columns if 'date' in col.lower()]
            
            if date_cols:
                return df[date_cols[0]].unique()
            else:
                raise ValueError("Cannot find date information in DataFrame - no date column or datetime index")

    def get_orders(self):
        """
        Get the generated orders DataFrame.
        @return: DataFrame containing the generated orders.
        """
        return self.orders_df