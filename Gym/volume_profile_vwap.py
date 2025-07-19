import pandas as pd
import numpy as np
import logging
from typing import Dict, List, Optional, Tuple
from datetime import datetime, timedelta

logger = logging.getLogger(__name__)

class VolumeProfileVWAP:
    """
    VWAP baseline strategy that follows realistic intraday volume patterns.
    
    Key Features:
    - 1-minute granularity volume profiles
    - 21-day historical lookback (or available data)
    - TWAP fallback for insufficient data (< 1 day)
    - Risk management: 4x TWAP cap per bin
    """
    
    def __init__(self, stock_df_list: Dict[str, pd.DataFrame], lookback_days: int = 21):
        """
        Initialize VolumeProfileVWAP with historical stock data.
        
        @param stock_df_list: Dictionary mapping ticker -> DataFrame with OHLCV data
        @param lookback_days: Number of days to look back for volume profile (default 21)
        """
        self.stock_df_list = stock_df_list
        self.lookback_days = lookback_days
        self.volume_profiles = {}
        self.twap_profiles = {}
        self.min_data_threshold = 1  # Minimum days of data required
        
        logger.info(f"Initializing VolumeProfileVWAP with {lookback_days}-day lookback")
        self._build_all_volume_profiles()
    
    def _build_all_volume_profiles(self):
        """Build volume profiles for all tickers in the dataset."""
        for ticker, df in self.stock_df_list.items():
            try:
                self._build_ticker_volume_profile(ticker, df)
            except Exception as e:
                logger.warning(f"Failed to build volume profile for {ticker}: {e}, using synthetic profile")
                # Fallback to synthetic profile for this ticker
                profile = self._create_synthetic_volume_profile()
                profile = self._apply_risk_caps(profile)
                self.volume_profiles[ticker] = profile
    
    def _build_ticker_volume_profile(self, ticker: str, df: pd.DataFrame):
        """
        Build 1-minute intraday volume profile for a specific ticker.
        
        @param ticker: Stock ticker symbol
        @param df: DataFrame with historical OHLCV data
        """
        if len(df) < self.min_data_threshold:
            logger.info(f"Insufficient data for {ticker}: {len(df)} days < {self.min_data_threshold} day minimum, using synthetic profile")
            profile = self._create_synthetic_volume_profile()
            profile = self._apply_risk_caps(profile)
            self.volume_profiles[ticker] = profile
            logger.debug(f"Built synthetic volume profile for {ticker}: {len(profile)} minute bins")
            return
        
        # Ensure Date column is datetime
        if 'Date' in df.columns:
            df = df.copy()
            df['Date'] = pd.to_datetime(df['Date'])
            df = df.sort_values('Date')
        
        # Use last N days for profile
        recent_df = df.tail(min(self.lookback_days * 390, len(df)))  # 390 minutes per trading day
        
        if len(recent_df) < 390:  # Less than 1 full trading day
            logger.info(f"Insufficient minute data for {ticker}: {len(recent_df)} < 390, using synthetic profile")
            profile = self._create_synthetic_volume_profile()
            profile = self._apply_risk_caps(profile)
            self.volume_profiles[ticker] = profile
            logger.debug(f"Built synthetic volume profile for {ticker}: {len(profile)} minute bins")
            return
        
        # Extract minute-of-day from index (assuming minute-level data)
        # If data is daily, we'll create a synthetic intraday profile
        if len(recent_df) < len(df) * 100:  # Likely daily data, not minute data
            logger.info(f"Daily data detected for {ticker}, creating synthetic intraday volume profile")
            profile = self._create_synthetic_volume_profile()
        else:
            # Minute-level data available
            profile = self._calculate_historical_volume_profile(recent_df)
        
        # Apply risk management: cap at 4x TWAP
        profile = self._apply_risk_caps(profile)
        
        self.volume_profiles[ticker] = profile
        logger.debug(f"Built volume profile for {ticker}: {len(profile)} minute bins")
    
    def _calculate_historical_volume_profile(self, df: pd.DataFrame) -> np.ndarray:
        """
        Calculate historical intraday volume profile from minute-level data.
        
        @param df: DataFrame with minute-level data
        @return: Normalized volume profile (390 minutes, sums to 1.0)
        """
        try:
            # Ensure we have a datetime index or Date column
            if 'Date' in df.columns:
                df = df.copy()
                df['Date'] = pd.to_datetime(df['Date'])
                
                # If no time component, assume this is daily data
                if df['Date'].dt.time.nunique() <= 1:
                    logger.info("Daily data detected, falling back to synthetic profile")
                    return self._create_synthetic_volume_profile()
                
                df.set_index('Date', inplace=True)
            elif isinstance(df.index, pd.DatetimeIndex):
                pass  # Already has datetime index
            else:
                logger.warning("No datetime information found, falling back to synthetic profile")
                return self._create_synthetic_volume_profile()
            
            # Check if we have Volume column
            if 'Volume' not in df.columns:
                logger.warning("No Volume column found, falling back to synthetic profile")
                return self._create_synthetic_volume_profile()
            
            # Extract time-of-day information
            df = df.copy()
            df['time'] = df.index.time
            df['date'] = df.index.date
            
            # Filter to regular trading hours (9:30 AM - 4:00 PM ET)
            # Convert to minutes from market open (9:30 AM = 0, 4:00 PM = 390)
            df['minute_of_day'] = (
                df.index.hour * 60 + df.index.minute - 570  # 570 = 9:30 AM in minutes
            )
            
            # Filter to trading hours (0-389 minutes)
            trading_df = df[(df['minute_of_day'] >= 0) & (df['minute_of_day'] < 390)].copy()
            
            if len(trading_df) < 100:  # Not enough intraday data
                logger.info("Insufficient intraday data, falling back to synthetic profile")
                return self._create_synthetic_volume_profile()
            
            # Group by minute-of-day and calculate average volume
            minute_volumes = trading_df.groupby('minute_of_day')['Volume'].mean()
            
            # Create full 390-minute profile (fill missing minutes with interpolation)
            full_profile = np.zeros(390)
            
            for minute in range(390):
                if minute in minute_volumes.index:
                    full_profile[minute] = minute_volumes[minute]
                else:
                    # Interpolate missing minutes
                    # Find nearest available minutes
                    available_minutes = minute_volumes.index.to_numpy()
                    if len(available_minutes) > 0:
                        nearest_idx = np.argmin(np.abs(available_minutes - minute))
                        full_profile[minute] = minute_volumes.iloc[nearest_idx]
                    else:
                        full_profile[minute] = np.mean(minute_volumes) if len(minute_volumes) > 0 else 1.0
            
            # Smooth the profile to reduce noise
            from scipy.ndimage import gaussian_filter1d
            try:
                full_profile = gaussian_filter1d(full_profile, sigma=2)
            except ImportError:
                # Fallback: simple moving average smoothing
                window = 5
                smoothed = np.zeros_like(full_profile)
                for i in range(len(full_profile)):
                    start = max(0, i - window // 2)
                    end = min(len(full_profile), i + window // 2 + 1)
                    smoothed[i] = np.mean(full_profile[start:end])
                full_profile = smoothed
            
            # Ensure no zero values (minimum 1% of mean)
            min_volume = np.mean(full_profile) * 0.01
            full_profile = np.maximum(full_profile, min_volume)
            
            # Normalize to sum to 1.0
            profile = full_profile / np.sum(full_profile)
            
            logger.debug(f"Built historical volume profile: min={np.min(profile):.6f}, max={np.max(profile):.6f}")
            
            return profile
            
        except Exception as e:
            logger.warning(f"Error calculating historical volume profile: {e}, falling back to synthetic")
            return self._create_synthetic_volume_profile()
    
    def _create_synthetic_volume_profile(self) -> np.ndarray:
        """
        Create a synthetic U-shaped intraday volume profile.
        
        @return: Normalized volume profile (390 minutes, sums to 1.0)
        """
        minutes = 390  # 6.5 hours * 60 minutes
        
        # Create U-shaped profile: high at start/end, low in middle
        x = np.linspace(0, 1, minutes)
        
        # Combination of opening surge, lunch lull, and closing surge
        opening_surge = np.exp(-10 * x)  # Exponential decay from open
        closing_surge = np.exp(-10 * (1 - x))  # Exponential rise to close
        lunch_lull = 1 - 0.5 * np.exp(-50 * (x - 0.5)**2)  # Reduced volume at midday
        
        # Combine components
        profile = (opening_surge + closing_surge + 0.3) * lunch_lull
        
        # Normalize to sum to 1.0
        profile = profile / np.sum(profile)
        
        return profile
    
    def _apply_risk_caps(self, profile: np.ndarray) -> np.ndarray:
        """
        Apply risk management: cap each bin at 4x TWAP allocation.
        
        @param profile: Original volume profile
        @return: Risk-adjusted profile
        """
        twap_allocation = 1.0 / len(profile)  # Equal allocation for TWAP
        max_allocation = 4.0 * twap_allocation  # 4x TWAP cap
        
        # Cap excessive allocations
        capped_profile = np.minimum(profile, max_allocation)
        
        # Redistribute excess volume proportionally to uncapped bins
        excess_volume = np.sum(profile) - np.sum(capped_profile)
        
        if excess_volume > 0:
            # Find bins that aren't capped
            uncapped_mask = capped_profile < max_allocation
            if np.any(uncapped_mask):
                # Redistribute excess proportionally to uncapped bins
                uncapped_total = np.sum(capped_profile[uncapped_mask])
                if uncapped_total > 0:
                    redistribution = excess_volume * (capped_profile[uncapped_mask] / uncapped_total)
                    capped_profile[uncapped_mask] += redistribution
        
        # Renormalize to ensure sum = 1.0
        capped_profile = capped_profile / np.sum(capped_profile)
        
        return capped_profile
    
    def _is_synthetic_profile(self, profile: np.ndarray) -> bool:
        """
        Heuristic to determine if a profile is synthetic or historical.
        Synthetic profiles have smoother, more predictable patterns.
        
        @param profile: Volume profile to analyze
        @return: True if likely synthetic, False if likely historical
        """
        # Check for extremely smooth patterns typical of synthetic profiles
        # Calculate second derivative to measure smoothness
        if len(profile) < 3:
            return True
        
        second_derivative = np.diff(profile, n=2)
        smoothness = np.std(second_derivative)
        
        # Synthetic profiles tend to be much smoother
        # This threshold may need tuning based on real data characteristics
        smoothness_threshold = 0.0001
        
        return smoothness < smoothness_threshold
    
    def generate_execution_schedule(self, ticker: str, order_qty: int, time_horizon: int, 
                                  start_time: int = 0) -> np.ndarray:
        """
        Generate execution schedule for a specific order following volume profile.
        
        @param ticker: Stock ticker
        @param order_qty: Total shares to execute
        @param time_horizon: Number of minutes for execution
        @param start_time: Starting minute (default 0)
        @return: Array of shares to execute each minute
        """
        # Get volume profile for this ticker
        profile = self.volume_profiles.get(ticker)
        
        if profile is None:
            # This shouldn't happen anymore, but fallback to TWAP just in case
            logger.warning(f"No volume profile found for {ticker}, using TWAP fallback")
            schedule = np.full(time_horizon, order_qty / time_horizon)
        else:
            # Use volume profile, scaled to time horizon
            if time_horizon <= len(profile):
                # Take subset of profile
                used_profile = profile[start_time:start_time + time_horizon]
            else:
                # Repeat/extend profile if horizon is longer
                repeats = int(np.ceil(time_horizon / len(profile)))
                extended_profile = np.tile(profile, repeats)
                used_profile = extended_profile[:time_horizon]
            
            # Normalize and scale to order quantity
            used_profile = used_profile / np.sum(used_profile)
            schedule = used_profile * order_qty
        
        return schedule
    
    def execute_order(self, ticker: str, order_qty: int, time_horizon: int, 
                     start_time: int = 0) -> List[Dict]:
        """
        Execute a single order using volume profile strategy.
        
        @param ticker: Stock ticker
        @param order_qty: Total shares to execute
        @param time_horizon: Number of minutes for execution
        @param start_time: Starting minute
        @return: List of trade information dictionaries
        """
        schedule = self.generate_execution_schedule(ticker, order_qty, time_horizon, start_time)
        
        trades = []
        cumulative_qty = 0
        
        for minute, qty in enumerate(schedule):
            if qty > 0:
                cumulative_qty += qty
                trades.append({
                    'minute': start_time + minute,
                    'shares': qty,
                    'cumulative_shares': cumulative_qty,
                    'strategy': 'volume_profile_vwap'
                })
        
        return trades
    
    def get_volume_proportion(self, ticker: str, minute: int) -> float:
        """
        Get expected volume proportion at a specific minute.
        
        @param ticker: Stock ticker
        @param minute: Minute of day (0-389)
        @return: Volume proportion (0.0 to 1.0)
        """
        profile = self.volume_profiles.get(ticker)
        
        if profile is None:
            # This shouldn't happen anymore, but fallback to TWAP just in case
            logger.warning(f"No volume profile found for {ticker}, using TWAP proportion")
            return 1.0 / 390
        
        minute_idx = minute % len(profile)
        return profile[minute_idx]
    
    def get_ticker_stats(self, ticker: str) -> Dict:
        """
        Get statistics about volume profile for a ticker.
        
        @param ticker: Stock ticker
        @return: Dictionary with profile statistics
        """
        profile = self.volume_profiles.get(ticker)
        
        if profile is None:
            # This shouldn't happen anymore, but provide fallback stats just in case
            logger.warning(f"No volume profile found for {ticker}, returning TWAP stats")
            return {
                'strategy': 'TWAP_fallback',
                'profile_length': 390,
                'max_allocation': 1.0/390,
                'min_allocation': 1.0/390
            }
        
        # Determine if this is synthetic or historical based on the profile characteristics
        # Synthetic profiles have specific patterns we can detect
        is_synthetic = self._is_synthetic_profile(profile)
        
        return {
            'strategy': 'synthetic_profile' if is_synthetic else 'historical_profile',
            'profile_length': len(profile),
            'max_allocation': np.max(profile),
            'min_allocation': np.min(profile),
            'peak_minute': np.argmax(profile),
            'mean_allocation': np.mean(profile)
        } 