# class which create orders for the gym
import pandas as pd
import numpy as np
import random
import mkt_data_yfinance as mdy
import logging

# Set up logger for this module
logger = logging.getLogger(__name__)


class OrderGenerator:
    def __init__(self, stock_df_list, market_data, num_orders=10,  min_adv_pct=0.0001,
                 max_adv_pct=0.25, min_time_horizon=2, max_time_horizon=390, sd_delta=0, 
                 ed_delta=0, debug=False, seed=None):
        """
        Initialize the OrderGenerator with a random DataFrame of orders.
        @param stock_df_list: List of DataFrames containing stock data.
        @param market_data: Instance of MarketDataLoader to fetch market data.
        @param num_orders: Number of random orders to generate.
        @param min_adv_pct: Minimum percentage of ADV for orders.
        @param max_adv_pct: Maximum percentage of ADV for orders.
        @param min_time_horizon: Minimum time horizon for orders in minutes.
        @param max_time_horizon: Maximum time horizon for orders in minutes.
        @param sd_delta: Start date delta - number of days to shift the start date of the order generation.
        @param ed_delta: End date delta - number of days to shift the end date of the order generation.
        @param debug: Enable debug logging.
        @param seed: Random seed for reproducibility.
        @return: None
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
        self.orders_df = self._generate_orders()

    def _generate_orders(self):
        """
        Generate a dataframe of random orders based on the stock data.
        @return: DataFrame containing the generated orders.
        """
        # extract unique stock names from the list of stock data frames 
        stocks = self.mdl.sample_tickers_from_data(self.stock_df_list, self.num_orders)
        logger.debug(f"Generating {self.num_orders} orders for stocks: {stocks}")
        
        # calculate average daily volume (ADV) for each stock
        advs = self._calculate_advs()
        logger.debug(f"Calculated ADV for stocks: {advs}")

        # Get date range from the data
        # Use the first stock to determine the date range (assuming all stocks have same date range)
        sample_ticker = list(self.stock_df_list.keys())[0]
        sample_df = self.stock_df_list[sample_ticker]
        
        # Get unique dates from the data
        unique_dates = pd.to_datetime(sample_df.index.date).unique()
        unique_dates = sorted(unique_dates)
        
        # Apply sd_delta and ed_delta to filter dates
        if self.sd_delta > 0:
            unique_dates = unique_dates[self.sd_delta:]
        if self.ed_delta > 0:
            unique_dates = unique_dates[:-self.ed_delta]
            
        if len(unique_dates) == 0:
            raise ValueError(f"No dates available after applying sd_delta={self.sd_delta} and ed_delta={self.ed_delta}")
            
        logger.debug(f"Available dates after filtering: {len(unique_dates)} days from {unique_dates[0]} to {unique_dates[-1]}")
        
        # Calculate orders per day for even distribution
        orders_per_day = self.num_orders // len(unique_dates)
        remaining_orders = self.num_orders % len(unique_dates)
        
        logger.debug(f"Distributing {self.num_orders} orders across {len(unique_dates)} days: {orders_per_day} per day with {remaining_orders} extra")
        
        # Initialize lists to store order data
        order_dates = []
        order_start_times = []
        order_end_times = []
        order_time_horizons = []
        order_stocks = []  # Track which stocks were actually used
        
        order_idx = 0
        for date_idx, order_date in enumerate(unique_dates):
            # Calculate number of orders for this date
            num_orders_this_date = orders_per_day
            # Distribute remaining orders across first few days
            if date_idx < remaining_orders:
                num_orders_this_date += 1
                
            logger.debug(f"Generating {num_orders_this_date} orders for date {order_date}")
            
            for _ in range(num_orders_this_date):
                if order_idx >= self.num_orders:
                    break
                    
                # Get the ticker for this order
                ticker = stocks[order_idx]
                df = self.stock_df_list[ticker]
                
                # Filter data for the selected date
                date_data = df[df.index.date == pd.to_datetime(order_date).date()]
                available_minutes = len(date_data)
                
                if available_minutes == 0:
                    logger.warning(f"No data available for ticker {ticker} on date {order_date}, skipping")
                    order_idx += 1  # Still increment to try next ticker
                    continue
                
                # Generate time horizon
                time_horizon = np.random.randint(self.min_time_horizon, 
                                               min(self.max_time_horizon, available_minutes) + 1)
                
                # Generate start time ensuring order can complete within the day
                max_start = max(0, available_minutes - time_horizon)
                start_time = np.random.randint(0, max_start + 1) if max_start > 0 else 0
                end_time = start_time + time_horizon
                
                order_dates.append(order_date)
                order_time_horizons.append(time_horizon)
                order_start_times.append(start_time)
                order_end_times.append(end_time)
                order_stocks.append(ticker)
                
                order_idx += 1

        # Get the actual number of orders generated
        actual_num_orders = len(order_dates)
        logger.debug(f"Actually generated {actual_num_orders} orders out of {self.num_orders} requested")
        
        # Convert to numpy arrays
        time_horizons = np.array(order_time_horizons)
        start_times = np.array(order_start_times)
        end_times = np.array(order_end_times)

        # scale to rational size for horizon given
        pct_of_day = time_horizons / 390.0
        logger.debug(f"min_adv_percent: {self.min_adv_pct}, max_adv_percent: {self.max_adv_pct}, Percentage of day for time horizon: {pct_of_day}")
        min_ehv_pct = self.min_adv_pct * pct_of_day
        max_ehv_pct = self.max_adv_pct * pct_of_day
        
        # select random percentages of ADV for each order - use actual number of orders
        adv_pct = np.random.uniform(min_ehv_pct, max_ehv_pct, size=actual_num_orders)
        logger.debug(f"Selected ADV percentages: {adv_pct}")

        # compute fractional quantities using the stocks that were actually used
        fractional_qty_series = advs.reindex(order_stocks) * adv_pct
        
        # now np.round will work
        order_quantities = (fractional_qty_series.round(0).astype(int).to_numpy())  

        # Log before rounding to see the fractional quantities
        logger.debug(f"Calculated fractional order quantities: {order_quantities}")


        # Check for zero quantities
        if (order_quantities == 0).any():
            zero_indices = np.where(order_quantities == 0)[0]
            logger.warning(f"Found {len(zero_indices)} orders with quantity 0 after rounding.")
            for i in zero_indices:
                stock_ticker = order_stocks[i]
                original_qty = order_quantities[i] * adv_pct[i]
                logger.warning(f"  - Order for ticker {stock_ticker}:")
                logger.warning(f"    - ADV: {order_quantities[i]:.2f}, ADV_pct: {adv_pct[i]:.6f}")
                logger.warning(f"    - Fractional Qty before rounding: {original_qty:.4f}")
        

        # select random sides (buy/sell) - use actual number of orders
        sides = np.random.choice(['buy', 'sell'], size=actual_num_orders)
        
        # create a DataFrame with the generated orders
        orders_df = pd.DataFrame({
            'ticker': order_stocks,
            'order_qty': order_quantities,
            'adv_pct': adv_pct, # percentage of ADV for each order
            'ehv_pct': adv_pct/pct_of_day, # percentage of expected horizon volume (EHV)
            'adv': advs.reindex(order_stocks).to_numpy(),
            'date': order_dates,  # Add the date field
            'start_time': start_times,
            'end_time': end_times,
            'time_horizon': time_horizons,
            'side': sides
        })
        
        # Add today's volatility and intra-order return for each order
        for i, (ticker, date) in enumerate(zip(order_stocks, order_dates)):
            df = self.stock_df_list[ticker]
            # Filter data for the specific date
            date_data = df[df.index.date == pd.to_datetime(date).date()]
            
            # Calculate intra-order return (from start to end of the order)
            start_price = date_data.iloc[start_times[i]]['Open']
            end_price = date_data.iloc[end_times[i] - 1]['Close']  # -1 because end_time is exclusive
            intra_return = (end_price - start_price) / start_price
            orders_df.loc[i, 'intra_order_return'] = intra_return
            # Use existing DailyVol from market data for that date
            orders_df.loc[i, 'today_volatility'] = date_data['DailyVol'].iloc[0]
        
        logger.debug(f"Generated orders DataFrame:\n{orders_df}")
        logger.debug(f"Orders per date distribution:\n{orders_df['date'].value_counts().sort_index()}")
        return orders_df
    

    def _calculate_advs(self):
        """
        Calculate the average daily volume (ADV) for each stock in the stock_df_list.
        @return: Series containing the ADV for each stock.
        """
        
        advs = {}
        for ticker, df in self.stock_df_list.items():
            # sum the volume, group by date, and take the mean
            daily_volume = df['Volume'].groupby(df.index.date).sum()
            adv = daily_volume.mean()
            advs[ticker] = adv
            
        return pd.Series(advs)
    

    def get_orders(self):
        """
        Get the generated orders DataFrame.
        @return: DataFrame containing the generated orders.
        """
        return self.orders_df