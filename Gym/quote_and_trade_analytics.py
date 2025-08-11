"""
QuoteAndTradeAnalytics: Centralized class for loading and accessing daily analytics data.

This class handles loading daily analytics from parquet files and provides
convenient methods for accessing analytics data throughout the project.
"""

import os
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Union
from datetime import datetime, date
import logging

logger = logging.getLogger(__name__)


class QuoteAndTradeAnalytics:
    """
    Centralized analytics class for loading and accessing daily market analytics.
    
    This class loads pre-computed analytics (ADV, spreads, depth, volatility) 
    and provides convenient access methods for use across the project.
    """
    
    def __init__(self, analytics_path: str = "mana_data/daily_analytics_parquet", 
                 symbols: Optional[List[str]] = None,
                 load_immediately: bool = True):
        """
        Initialize the analytics class.
        @param analytics_path: Path to the analytics parquet files
        @param symbols: Optional list of symbols to load (if None, loads all available)
        @param load_immediately: Whether to load analytics data immediately
        """
        self.analytics_path = analytics_path
        self.target_symbols = symbols
        self.analytics_data: Dict[str, pd.DataFrame] = {}
        self.available_columns = []
        self.date_ranges = {}
        
        if load_immediately:
            self.load_analytics()
    
    def load_analytics(self) -> None:
        """
        Load analytics data from parquet files.
        """
        logger.info(f"Loading analytics from: {self.analytics_path}")
        
        if not os.path.exists(self.analytics_path):
            logger.warning(f"Analytics path does not exist: {self.analytics_path}")
            return
        
        try:
            # Get all symbol directories
            symbol_dirs = [d for d in os.listdir(self.analytics_path) if d.startswith('sym=')]
            logger.info(f"Found {len(symbol_dirs)} symbol directories")
            
            loaded_count = 0
            for symbol_dir in symbol_dirs:
                symbol = symbol_dir.split('=')[1]
                
                # Filter by target symbols if specified
                if self.target_symbols and symbol not in self.target_symbols:
                    continue
                
                symbol_path = os.path.join(self.analytics_path, symbol_dir)
                
                # Get all date directories for this symbol
                date_dirs = [d for d in os.listdir(symbol_path) if d.startswith('date=')]
                
                if date_dirs:
                    all_dfs = []
                    for date_dir in date_dirs:
                        try:
                            # Load the analytics.parquet file from each date directory
                            file_path = os.path.join(symbol_path, date_dir, 'analytics.parquet')
                            if os.path.exists(file_path):
                                df = pd.read_parquet(file_path)
                                all_dfs.append(df)
                        except Exception as e:
                            logger.warning(f"Could not load analytics for {symbol} on {date_dir}: {e}")
                    
                    if all_dfs:
                        combined_df = pd.concat(all_dfs, ignore_index=True)
                        combined_df['date'] = pd.to_datetime(combined_df['date'])
                        combined_df = combined_df.sort_values('date').reset_index(drop=True)
                        
                        self.analytics_data[symbol] = combined_df
                        loaded_count += 1
                        
                        # Store date range info
                        self.date_ranges[symbol] = {
                            'start': combined_df['date'].min(),
                            'end': combined_df['date'].max(),
                            'count': len(combined_df)
                        }
                        
                        # Store available columns (from first successful load)
                        if not self.available_columns:
                            self.available_columns = [col for col in combined_df.columns if col != 'date']
                        
                        logger.debug(f"Loaded analytics for {symbol}: {len(combined_df)} records "
                                   f"from {combined_df['date'].min().date()} to {combined_df['date'].max().date()}")
                        
        except Exception as e:
            logger.error(f"Error loading analytics: {e}")
            
        logger.info(f"Successfully loaded analytics for {loaded_count} symbols")
        if self.available_columns:
            logger.info(f"Available analytics columns: {self.available_columns}")
    

    def get_analytics_for_symbol_date(self, symbol: str, target_date: Union[str, date, datetime]) -> Optional[Dict]:
        """
        Get analytics for a specific symbol and date.
        @param symbol: Stock symbol
        @param target_date: Date to get analytics for
        @return: Dictionary with analytics values or None if not found
        """
        if symbol not in self.analytics_data:
            return None
            
        analytics_df = self.analytics_data[symbol]
        
        # Convert target_date to date object for comparison
        if isinstance(target_date, str):
            target_date = pd.to_datetime(target_date).date()
        elif isinstance(target_date, datetime):
            target_date = target_date.date()
            
        date_mask = analytics_df['date'].dt.date == target_date
        
        if not date_mask.any():
            return None
            
        row = analytics_df[date_mask].iloc[0]
        return {
            'adv_21_days': row.get('adv_21_days', 0),
            'avg_spread_21_days': row.get('avg_spread_21_days', 0),
            'avg_trade_count_21_days': row.get('avg_trade_count_21_days', 0),
            'avg_depth_21_days': row.get('avg_depth_21_days', 0),
            'vwap': row.get('vwap', 0),
            'daily_volatility': row.get('daily_volatility', 0),
            'daily_volatility_lag1': row.get('daily_volatility_lag1', 0),
            'daily_volatility_5d': row.get('daily_volatility_5d', 0)
        }
    

    def get_adv_for_symbol_date(self, symbol: str, target_date: Union[str, date, datetime]) -> float:
        """
        Get adv_21_days for a specific symbol and date.
        @param symbol: Stock symbol
        @param target_date: Date to get adv_21_days for
        @return: adv_21_days value for that date or 0 if not found
        """
        analytics = self.get_analytics_for_symbol_date(symbol, target_date)
        if analytics:
            return analytics['adv_21_days']
        return 0.0
    

    def has_analytics_for_symbol(self, symbol: str) -> bool:
        """
        Check if analytics are available for a symbol.
        @param symbol: Stock symbol
        @return: True if analytics are available
        """
        return symbol in self.analytics_data
    

    def has_data(self) -> bool:
        """
        Check if analytics data is available.
        @return: True if analytics data is available
        """
        return len(self.analytics_data) > 0


    def get_analytics_bulk(self, symbol_date_pairs: List[tuple]) -> Dict[tuple, Dict]:
        """
        Bulk load analytics for multiple symbol-date pairs.
        @param symbol_date_pairs: List of (symbol, date) tuples
        @return: Dictionary mapping (symbol, date) to analytics dict
        """
        result = {}
    
        for symbol, target_date in symbol_date_pairs:
            analytics = self.get_analytics_for_symbol_date(symbol, target_date)
            if analytics:
                result[(symbol, target_date)] = analytics
    
        return result


    def get_adv_matrix(self, stocks: List[str], dates: List[date]) -> np.ndarray:
        """
        Get ADV matrix for a list of stocks and dates.
        Optimized to minimize repeated date conversions.
        @param stocks: List of stock symbols
        @param dates: List of dates
        @return: ADV matrix
        """
        if not self.has_data():
            logger.warning("No analytics data available")
            return np.zeros((len(stocks), len(dates)))
    
        # Create a matrix of zeros with the correct shape
        adv_matrix = np.zeros((len(stocks), len(dates)))
    
        # Convert dates once to avoid repeated conversions
        dates_normalized = []
        for d in dates:
            if isinstance(d, str):
                dates_normalized.append(pd.to_datetime(d).date())
            elif isinstance(d, datetime):
                dates_normalized.append(d.date())
            else:
                dates_normalized.append(d)
    
        # Fill the matrix with ADV values
        for i, stock in enumerate(stocks):
            if stock not in self.analytics_data:
                continue
            
            analytics_df = self.analytics_data[stock]
            if 'adv_21_days' not in analytics_df.columns:
                continue
        
            # Convert the dataframe dates once per stock
            df_dates = analytics_df['date'].dt.date
        
            for j, target_date in enumerate(dates_normalized):
                # Find matching date
                date_mask = df_dates == target_date
                if date_mask.any():
                    adv_matrix[i, j] = analytics_df[date_mask].iloc[0].get('adv_21_days', 0)
    
        return adv_matrix       
    

    def __len__(self) -> int:
        """
        Return number of symbols with analytics.
        @return: Number of symbols with analytics
        """
        return len(self.analytics_data)
    

    def __contains__(self, symbol: str) -> bool:
        """
        Check if symbol has analytics.
        @param symbol: Stock symbol
        @return: True if symbol has analytics
        """
        return symbol in self.analytics_data
    

    def __repr__(self) -> str:
        """
        String representation.
        @return: String representation of the object
        """
        return f"QuoteAndTradeAnalytics({len(self.analytics_data)} symbols, {len(self.available_columns)} metrics)" 