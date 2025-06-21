# Market data loader library using yahoo finance

import yfinance as yf
import random
import logging
import numpy as np
import pandas as pd

# Set up logger for this module
logger = logging.getLogger(__name__)

# TODO: cache data locally and check for cache before downloading

class MarketDataLoader:
    """A class to handle market data loading and processing using Yahoo Finance."""
    
    def __init__(self, debug=False):
        """
        Initialize the MarketDataLoader
        @param debug: Enable debug logging.
        """
        self.debug = debug
        
        if self.debug:
            logger.setLevel(logging.DEBUG)
        
        logger.debug("MarketDataLoader initialized")
    
    def load_data(self, tickers, horizon='7d', interval='1m', dropna=True):
        """
        Load market data for multiple tickers using yfinance.
        @param tickers: List of ticker symbols (e.g., ['AAPL', 'GOOGL']).
        @param horizon: Time period to fetch data for (e.g., '7d' for last 7 days).
        @param interval: Data interval (e.g., '1m' for 1-minute bars).
        @param dropna: Whether to drop rows with NaN values.
        @return: Dictionary of DataFrames, each keyed by ticker symbol.
        """
        data_df_list = {}
        
        for ticker in tickers:
            # Download data
            logger.debug(f"Downloading data for {ticker}")
            data_df_list[ticker] = yf.download(
                ticker,
                period=horizon,
                interval=interval
            )
            
            if dropna:
                data_df_list[ticker].dropna(inplace=True)
            
            logger.debug(f"Downloaded data for {ticker} of type "
                        f"{type(data_df_list[ticker])} with shape {data_df_list[ticker].shape}")
        
        return data_df_list
    
    def pre_process_data(self, data_df_list, fields=None):
        """
        Pre-process the loaded market data.
        @param data_df_list: Dictionary of DataFrames to process.
        @param fields: Specific fields to process or extract.
        @return: Processed dictionary of DataFrames.
        """
        # TODO: Add new analytics
        logger.debug(f"Pre-processing data for {len(data_df_list)} tickers")
        processed = {}
        
        for ticker, df in data_df_list.items():
            if fields:
                df = df[fields]

            # Convert all columns to float32 to save memory
            df = df.astype(np.float32)

            # Forward fill missing data (use modern ffill)
            df.ffill(inplace=True)

            # Handle MultiIndex columns if present
            if isinstance(df.columns, pd.MultiIndex):
                # Flatten the MultiIndex by taking the second level (the actual column names)
                df.columns = df.columns.get_level_values(0)

            # ------------------------------------------------------------------
            # 1) Per-minute geometric mid-price and returns
            # ------------------------------------------------------------------
            mid_price = (df['High'] + df['Low']) * 0.5

            # Log-returns are numerically more stable for small price moves
            df['LogRet'] = np.log(mid_price).diff().fillna(0.0)

            # ------------------------------------------------------------------
            # 2) Daily volatility σ using intra-day High/Low range per Parkinson-style estimator
            #    Here we approximate σ_d by the relative price range of the day:
            #    σ_d = (High_max - Low_min) / Low_min .  (dimensionless fraction)
            # ------------------------------------------------------------------
            day_idx = df.index.floor('D')

            # Handle timezone-aware index properly
            if df.index.tz is not None:
                df['date'] = df.index.date
            else:
                df['date'] = df.index.floor('D')

            # Group by date and calculate daily stats
            daily_stats = df.groupby('date').agg({'High': 'max', 'Low': 'min'}).astype(np.float32)

            daily_stats['DailyVol'] = (daily_stats['High'] - daily_stats['Low']) / daily_stats['Low'].clip(lower=1e-6)
            daily_stats['DailyVolLag1'] = daily_stats['DailyVol'].shift(1).bfill().fillna(0.0)
            daily_stats['DailyVol5d'] = daily_stats['DailyVol'].rolling(window=5, min_periods=1).mean()

            # Map back to minute data using proper dictionary mapping
            df['DailyVol'] = df['date'].map(daily_stats['DailyVol']).fillna(0.0).astype(np.float32)
            df['DailyVolLag1'] = df['date'].map(daily_stats['DailyVolLag1']).fillna(0.0).astype(np.float32)
            df['DailyVol5d'] = df['date'].map(daily_stats['DailyVol5d']).fillna(0.0).astype(np.float32)

            df.drop('date', axis=1, inplace=True)

            df['DailyVol'] = df['DailyVol'].fillna(0.0)
            df['DailyVolLag1'] = df['DailyVolLag1'].fillna(0.0)
            df['DailyVol5d'] = df['DailyVol5d'].fillna(0.0)

            processed[ticker] = df
        
        return processed
    
    def sample_tickers_from_data(self, data_df_list, num_samples=1):
        """
        Extract keys from data dictionary and randomly sample tickers with replacement.
        @param data_df_list: Dictionary of DataFrames keyed by ticker symbols.
        @param num_samples: Number of random samples to draw.
        @return: List of randomly selected ticker symbols.
        """
        # Extract all keys (ticker symbols) from the dictionary
        tickers = list(data_df_list.keys())
        
        # Randomly sample with replacement
        if num_samples == 1:
            # For single sample, use random.choice
            result = random.choice(tickers)
        else:
            # For multiple samples, use random.choices (with replacement)
            result = random.choices(tickers, k=num_samples)
        
        logger.debug(f"Sampled {num_samples} ticker(s): {result}")
        return result

    def get_first_date(self, data_df_list, stock_name):
        """
        Get the earliest date from the market data using the Datetime field
        @param data_df_list: Dictionary of DataFrames keyed by ticker symbols.
        @return: Earliest date as a pandas Timestamp.
        """
        if stock_name not in data_df_list:
            logger.error(f"Stock {stock_name} not found in data.")
            return None
        
        df = data_df_list[stock_name]
        first_date = df.index.min()
        
        if pd.isna(first_date):
            logger.error(f"No valid date found for stock {stock_name}.")
            return None
        
        logger.debug(f"First date for {stock_name}: {first_date}")
        return first_date