# class which create orders for the gym
import pandas as pd
import numpy as np
import mkt_data_yfinance as mdy
import logging

# Set up logger for this module
logger = logging.getLogger(__name__)


class OrderGenerator:
    def __init__(self, stock_df_list, market_data, num_orders=10,  min_adv_pct=0.0001,
                 max_adv_pct=0.25, min_time_horizon=2, max_time_horizon=390, sd_delta=0, 
                 ed_delta=0, debug=False):
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
        @return: None
        """
        # Set logging level based on debug parameter
        if debug:
            logger.setLevel(logging.DEBUG)

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

        # shift the start and end dates by the specified deltas

        # select random time horizons between min_time_horizon and max_time_horizon
        time_horizons = np.random.randint(self.min_time_horizon,
                                          self.max_time_horizon,
                                          size=self.num_orders)

        # create randome start times for each order from 0..max_time_horizon-horizon
        start_times = np.random.randint(0, self.max_time_horizon - time_horizons, size=self.num_orders)
        end_times = start_times + time_horizons

        # scale to rational size for horizon given
        pct_of_day = time_horizons / 390.0
        logger.debug(f"min_adv_percent: {self.min_adv_pct}, max_adv_percent: {self.max_adv_pct}, Percentage of day for time horizon: {pct_of_day}")
        min_ehv_pct = self.min_adv_pct * pct_of_day
        max_ehv_pct = self.max_adv_pct * pct_of_day
        
        # select random percentages of ADV for each order
        adv_pct = np.random.uniform(min_ehv_pct, max_ehv_pct, size=self.num_orders)
        logger.debug(f"Selected ADV percentages: {adv_pct}")

        # compute fractional quantities
        fractional_qty_series = advs.reindex(stocks) * adv_pct
        
        # now np.round will work
        order_quantities = (fractional_qty_series.round(0).astype(int).to_numpy())  

        # Log before rounding to see the fractional quantities
        logger.debug(f"Calculated fractional order quantities: {order_quantities}")


        # Check for zero quantities
        if (order_quantities == 0).any():
            zero_indices = np.where(order_quantities == 0)[0]
            logger.warning(f"Found {len(zero_indices)} orders with quantity 0 after rounding.")
            for i in zero_indices:
                stock_ticker = stocks[i]
                original_qty = order_quantities[i] * adv_pct[i]
                logger.warning(f"  - Order for ticker {stock_ticker}:")
                logger.warning(f"    - ADV: {order_quantities[i]:.2f}, ADV_pct: {adv_pct[i]:.6f}")
                logger.warning(f"    - Fractional Qty before rounding: {original_qty:.4f}")
        

        # select random sides (buy/sell)
        sides = np.random.choice(['buy', 'sell'], size=self.num_orders)
        
        # create a DataFrame with the generated orders
        orders_df = pd.DataFrame({
            'ticker': stocks,
            'order_qty': order_quantities,
            'adv_pct': adv_pct,
            'adv': advs.reindex(stocks).to_numpy(),
            'start_time': start_times,
            'end_time': end_times,
            'time_horizon': time_horizons,
            'side': sides
        })
        
        # Add today's volatility and intra-order return for each order
        for i, ticker in enumerate(stocks):
            df = self.stock_df_list[ticker]
            # Calculate intra-order return (from start to end of the order)
            start_price = df.iloc[start_times[i]]['Open']
            end_price = df.iloc[end_times[i]]['Close']
            intra_return = (end_price - start_price) / start_price
            orders_df.loc[i, 'intra_order_return'] = intra_return
            # Use existing DailyVol from market data
            orders_df.loc[i, 'today_volatility'] = df['DailyVol'].iloc[0]
        
        logger.debug(f"Generated orders DataFrame:\n{orders_df}")
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