"""
Propagator Parameter Loader Library

This library loads Y and tau parameters from parquet files organized by symbol and date,
and integrates them into the trading simulation environment. The parameters come from
a calibrated price propagator model that describes how market impact decays over time.

Y (immediate impact): The immediate price impact coefficient
tau (decay constant): Time constant for exponential decay of residual impact

Usage:
    from propagator_param_loader import PropagatorParamLoader
    
    loader = PropagatorParamLoader("mana_data/propagator_model_data")
    y, tau = loader.get_params("AAPL", "2022-07-05")
    
"""

import os
import pandas as pd
import polars as pl
import numpy as np
import logging
from typing import Dict, Tuple, Optional, Union, List
from datetime import datetime, date
import warnings

# Set up logger for this module
logger = logging.getLogger(__name__)


class PropagatorParamLoader:
    """
    Loader for Y and tau parameters from calibrated propagator model data.
    
    This class loads propagator parameters (Y, tau) that have been pre-calibrated 
    for different stocks and dates. These parameters are used to model how market 
    impact propagates and decays over time.
    """
    
    def __init__(self, data_dir: str = "mana_archive/propagator_model_data_cross_validation",
            date_format: str = "%Y%m%d"):
        """
        Initialize the propagator parameter loader.
        
        Args:
            data_dir: Root directory containing the propagator parameter data
            date_format: Format string for date parsing (default matches your setup)
        """
        self.data_dir = data_dir
        self.date_format = date_format
        self.df_params = None

        # Validate data directory exists
        if not os.path.exists(data_dir):
            raise FileNotFoundError(f"Propagator data directory not found: {data_dir}")
        
        # Get available symbols
        self.available_symbols = self._get_available_symbols()
        logger.info(f"PropagatorParamLoader initialized with {len(self.available_symbols)} symbols")
    
    def _get_available_symbols(self) -> List[str]:
        """Get list of available symbols in the data directory."""
        symbols = []
        try:
            for item in os.listdir(self.data_dir):
                symbol_path = os.path.join(self.data_dir, item)
                if os.path.isdir(symbol_path) and not item.startswith('.'):
                    # Handle sym=SYMBOL format
                    if item.startswith('sym='):
                        symbol = item.removeprefix('sym=')
                        symbols.append(symbol)
                    else:
                        # Fallback for old format
                        symbols.append(item)
        except Exception as e:
            logger.warning(f"Error reading symbols from {self.data_dir}: {e}")
        
        return sorted(symbols)
    
    def _normalize_date(self, date_input: Union[str, date, datetime]) -> str:
        """
        Normalize date input to string format used by the data files.
        
        Args:
            date_input: Date as string, date object, or datetime object
            
        Returns:
            Date string in the format expected by data files
        """
        if isinstance(date_input, str):
            # Try to parse and reformat to ensure consistency
            dt = datetime.strptime(date_input, self.date_format)
            return dt.strftime(self.date_format)
        
        elif isinstance(date_input, (date, datetime)):
            return date_input.strftime(self.date_format)
        
        else:
            raise TypeError(f"Invalid date type: {type(date_input)}")
    
    def _get_file_path(self, symbol: str, date_str: str) -> str:
        """
        Get the file path for a specific symbol and date.
        
        Args:
            symbol: Stock symbol (e.g., 'AAPL')
            date_str: Date string in format YYYYMMDD
            
        Returns:
            Full path to the parquet file
        """
        return os.path.join(
            self.data_dir, 
            f"sym={symbol}", 
            f"date={date_str}", 
            "daily_params.parquet"
        )
    
    def _load_params_from_file(self, symbol: str, date_str: str, 
                              y_column: str = 'Y_best', tau_column: str = 'tau_best') -> Tuple[float, float]:
        """
        Load Y and tau parameters from parquet file.
        
        Args:
            symbol: Stock symbol
            date_str: Date string in YYYYMMDD format
            y_column: Name of the Y (immediate impact) column
            tau_column: Name of the tau (decay rate) column
            
        Returns:
            Tuple of (Y, tau) parameters
            
        Raises:
            FileNotFoundError: If parameter file doesn't exist
            ValueError: If parameters are invalid or missing
        """
        file_path = self._get_file_path(symbol, date_str)
        
        if not os.path.exists(file_path):
            raise FileNotFoundError(
                f"Parameter file not found: {file_path}\n"
                f"Available symbols: {self.available_symbols[:10]}..."
            )
        
        try:
            # Read parquet file using polars - load all columns and rows
            df = pl.read_parquet(file_path)
            
            if df.is_empty():
                raise ValueError(f"Empty parameter file: {file_path}")
            
            # Check if required columns exist
            if y_column not in df.columns:
                raise ValueError(f"Column '{y_column}' not found in {file_path}. Available columns: {df.columns}")
            
            if tau_column not in df.columns:
                raise ValueError(f"Column '{tau_column}' not found in {file_path}. Available columns: {df.columns}")
            
            # Extract Y and tau from first row (typically one row per file)
            Y = df[y_column][0]
            tau = df[tau_column][0]
            
            # Convert to Python float and handle nulls
            if Y is None or pd.isna(Y):
                raise ValueError(f"Invalid Y parameter for {symbol} on {date_str} using column {y_column}: {Y}")
            
            if tau is None or pd.isna(tau):
                raise ValueError(f"Invalid tau parameter for {symbol} on {date_str} using column {tau_column}: {tau}")
            
            Y = float(Y)
            tau = float(tau)
            
            # Additional validation
            if tau <= 0:
                logger.warning(f"Non-positive tau in column {tau_column} for {symbol} on {date_str}: {tau}")
                tau = 1.0  # Default fallback
            
            if abs(Y) > 1.0:  # Sanity check for unreasonable Y values
                logger.warning(f"Large Y value for {symbol} on {date_str} using column {y_column}: {Y}")
            
            return Y, tau
            
        except Exception as e:
            logger.error(f"Error loading parameters from {file_path}: {e}")
            raise
    
    def get_params(self, symbol: str, date_input: Union[str, date, datetime], 
                   fallback_days: int = 5, y_column: str = 'Y_best', tau_column: str = 'tau_best') -> Tuple[float, float]:
        """
        Get Y and tau parameters for a specific symbol and date.
        
        Args:
            symbol: Stock symbol (e.g., 'AAPL')
            date_input: Date as string, date object, or datetime object
            fallback_days: Number of days to search backward if exact date not found
            y_column: Name of the Y (immediate impact) column (default: 'Y_best')
            tau_column: Name of the tau (decay rate) column (default: 'tau_best')
            
        Returns:
            Tuple of (Y, tau) parameters
            
        Raises:
            FileNotFoundError: If no parameters found within fallback window
        """
        symbol = symbol.upper()  # Normalize symbol
        date_str = self._normalize_date(date_input)
        
        # Try to load exact date
        try:
            params = self._load_params_from_file(symbol, date_str, y_column, tau_column)
            return params
        
        except FileNotFoundError:
            # Try fallback dates (search backwards)
            target_date = datetime.strptime(date_str, self.date_format)
            
            for days_back in range(1, fallback_days + 1):
                fallback_date = target_date - pd.Timedelta(days=days_back)
                fallback_date_str = fallback_date.strftime(self.date_format)
                
                try:
                    params = self._load_params_from_file(symbol, fallback_date_str, y_column, tau_column)
                    logger.info(f"Using fallback date {fallback_date_str} for {symbol} on {date_str}")
                    return params
                
                except FileNotFoundError:
                    continue
            
            # If no fallback found, raise error
            raise FileNotFoundError(
                f"No parameters found for {symbol} on {date_str} "
                f"(searched {fallback_days} days back)"
            )
    
    def get_params_batch(self, symbol_date_pairs: List[Tuple[str, Union[str, date, datetime]]], 
                        fallback_days: int = 5, y_column: str = 'Y_best', tau_column: str = 'tau_best') -> Dict[Tuple[str, str], Tuple[float, float]]:
        """
        Get parameters for multiple (symbol, date) pairs efficiently.
        
        Args:
            symbol_date_pairs: List of (symbol, date) tuples
            fallback_days: Number of days to search backward if exact date not found
            y_column: Name of the Y (immediate impact) column (default: 'Y_best')
            
            tau_column: Name of the tau (decay rate) column (default: 'tau_best')
            
        Returns:
            Dictionary mapping (symbol, normalized_date) to (Y, tau) parameters
        """
        results = {}
        
        if len(symbol_date_pairs) > 0:
            sample_symbol, sample_date = symbol_date_pairs[0]
            sample_date_str = self._normalize_date(sample_date)
            sample_path = self._get_file_path(sample_symbol, sample_date_str)
            logger.info(f"  Sample path would be: {sample_path}")
            logger.info(f"  Sample path exists: {os.path.exists(sample_path)}")
            logger.info(f"  Symbol_date_pairs: {symbol_date_pairs[0]} to {symbol_date_pairs[-1]}")
        
        found_count = 0
        missing_count = 0
        
        for symbol, date_input in symbol_date_pairs:
            try:
                normalized_date = self._normalize_date(date_input)
                params = self.get_params(symbol, date_input, fallback_days, y_column, tau_column)
                results[(symbol.upper(), normalized_date)] = params
                found_count += 1
            
            except Exception as e:
                missing_count += 1
                if missing_count <= 3:  # Only log first few failures
                    logger.debug(f"Failed to load parameters for {symbol} on {date_input}: {e}")
                continue
        
        logger.info(f"PropagatorParamLoader: Loaded {found_count}/{len(symbol_date_pairs)} parameter sets ({missing_count} missing)")
        
        return results
