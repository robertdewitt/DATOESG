"""
Baseline strategies for optimal execution comparison.
These strategies provide simple benchmarks to compare against trained RL models.
"""

import numpy as np
import random
import logging

logger = logging.getLogger(__name__)

MINUTES_IN_DAY = 390.0

class RandomBaseline:
    """
    Baseline strategy that selects random actions.
    """
    
    def __init__(self, name="Random_Baseline", seed=None):
        self.name = name
        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)
        logger.info(f"Initialized {self.name} strategy")
    
    def predict(self, observation, deterministic=True):
        """
        Predict random action.
        
        @param observation: Current observation (unused)
        @param deterministic: Whether to use deterministic prediction (unused)
        @return: Tuple of (action, None) to match stable-baselines3 interface
        """
        # Action space has 11 actions (0-10)
        action = random.randint(0, 10)
        return action, None


class ConstantRateBaseline:
    """
    Baseline strategy that always trades at a constant rate.
    """
    
    def __init__(self, action_index=5, name="ConstantRate_Baseline"):
        """
        @param action_index: Index into action space (default 5 = fraction 0.0 = ADV rate)
        """
        self.action_index = action_index
        self.name = name
        logger.info(f"Initialized {self.name} strategy with action index {action_index}")
    
    def predict(self, observation, deterministic=True):
        """
        Predict constant action.
        
        @param observation: Current observation (unused)
        @param deterministic: Whether to use deterministic prediction (unused)
        @return: Tuple of (action, None) to match stable-baselines3 interface
        """
        return self.action_index, None


class VWAPBaseline:
    """
    Adaptive VWAP that adjusts participation when execution deviates from target. 
    assumes a linear target which could be improved with volume profile based tracking.
    """
    
    def __init__(self, name="AdaptiveVWAP_Baseline", correction_factor=0.5):
        self.name = name
        self.correction_factor = correction_factor  # How aggressively to correct
        self.initial_shares = None
        self.initial_time_remaining = None
        self.cumulative_volume = 0.0
        self._action_values = np.array(
            [-1.0, -0.75, -0.5, -0.25, -0.1, 0.0, 0.1, 0.25, 0.5, 0.75, 1.0]
        )
        logger.info(f"Initialized {self.name} strategy")
    
    def predict(self, observation, deterministic=True):
        volume = float(observation[1])
        time_remaining = float(observation[2])
        shares_remaining = float(observation[3])
        
        if self.initial_shares is None:
            self.initial_shares = shares_remaining
        
        if time_remaining <= 1.0:
            return 5, None
        
        if self.initial_time_remaining is None:
            self.initial_time_remaining = time_remaining

        # Track execution progress
        shares_executed = self.initial_shares - shares_remaining
        completion_ratio = shares_executed / self.initial_shares if self.initial_shares > 0 else 0
        
        # Time progress
        time_elapsed = self.initial_time_remaining - time_remaining
        time_ratio = time_elapsed / self.initial_time_remaining if self.initial_time_remaining > 0 else 0
        
        # If we're behind schedule (completion < time progress), trade more aggressively
        deviation = completion_ratio - time_ratio
        
        # Adjust fraction based on how far behind/ahead we are add 5% for frotloading as it tends to backload due to no volume profiles
        required_fraction = -deviation * self.correction_factor + 0.1
        
        required_fraction = np.clip(required_fraction, -1.0, 1.0)
        action_idx = np.argmin(np.abs(self._action_values - required_fraction))
        
        return int(action_idx), None


class AdaptiveVWAPBaseline:
    """
    Adaptive VWAP baseline that adjusts trading rate based on market conditions.
    """
    
    def __init__(self, name="AdaptiveVWAP_Baseline"):
        self.name = name
        logger.info(f"Initialized {self.name} strategy")
    
    def predict(self, observation, deterministic=True):
        """
        Predict adaptive VWAP action based on current market volume and time pressure.
        
        @param observation: Current observation containing volume and time info
        @param deterministic: Whether to use deterministic prediction (unused)
        @return: Tuple of (action, None) to match stable-baselines3 interface
        """
        # Observation format: [mid_price, volume, time_remaining, shares_remaining, order_adv_pct, ...]
        if len(observation) >= 4:
            volume = observation[1]  # Current market volume
            time_remaining = observation[2]  # Time remaining
            shares_remaining = observation[3]  # Shares remaining
            
            # Adaptive logic based on volume and time pressure
            if time_remaining <= 1:
                # Very urgent - trade aggressively
                action = 9  # fraction 0.75 = 1.75 * ADV rate
            elif time_remaining <= 5:
                # Somewhat urgent - trade faster
                action = 7  # fraction 0.25 = 1.25 * ADV rate
            elif volume > 100000:  # High volume - can trade more
                action = 6  # fraction 0.1 = 1.1 * ADV rate
            elif volume < 10000:  # Low volume - trade less
                action = 4  # fraction -0.1 = 0.9 * ADV rate
            else:
                # Normal conditions - use baseline rate
                action = 5  # fraction 0.0 = ADV rate
        else:
            # Fallback to baseline ADV rate
            action = 5
            
        return action, None


class TWAPBaseline:
    def __init__(self, name="TWAP_Baseline"):
        self.name = name
        self.initial_shares = None
        self.initial_time_remaining = None
        self._action_values = np.array(
            [-1.0, -0.75, -0.5, -0.25, -0.1, 0.0, 0.1, 0.25, 0.5, 0.75, 1.0]
        )
        
    def predict(self, observation, deterministic=True):
        volume = float(observation[1])
        time_remaining = float(observation[2])
        shares_remaining = float(observation[3])
        ehv_pct = float(observation[5])
        
        # Initialize on first call
        if self.initial_shares is None:
            self.initial_shares = shares_remaining
            self.initial_time_remaining = time_remaining
        
        # TWAP target: q_t = Q_0 / H
        H = max(int(self.initial_time_remaining), 1)
        q_target = self.initial_shares / H
        
        # Last minute: complete order (action doesn't matter, env forces completion)
        if time_remaining <= 1.0:
            return 5, None
        
        # Convert target shares to required POV adjustment
        # The env uses: trade_size = ehv_pct * (1 + fraction) * volume
        # So: fraction = (q_target / (ehv_pct * volume)) - 1
        
        if volume > 0 and ehv_pct > 0:
            rem_pct_of_day = time_remaining / MINUTES_IN_DAY
            ehv_remaining = ehv_pct * rem_pct_of_day * volume
            if ehv_remaining > 0:
                required_fraction = (q_target / ehv_remaining) - 1.0
            else:
                required_fraction = 0.0
        else:
            required_fraction = 0.0
        
        # Clamp and find nearest action
        required_fraction = np.clip(required_fraction, -1.0, 1.0)
        action_idx = np.argmin(np.abs(self._action_values - required_fraction))
        
        return int(action_idx), None


class AlmgrenChrissBaseline:
    def __init__(self, name="AC_Baseline", risk_aversion=1e-6):
        self.name = name
        self.risk_aversion = risk_aversion
        self._action_values = np.array(
            [-1.0, -0.75, -0.5, -0.25, -0.1, 0.0, 0.1, 0.25, 0.5, 0.75, 1.0]
        )
        
    def predict(self, observation, deterministic=True):
        # Extract needed values
        mid_price = float(observation[0])
        volume = float(observation[1])
        time_remaining = float(observation[2])
        shares_remaining = float(observation[3])
        ehv_pct = float(observation[5])
        daily_vol = float(observation[14])  # daily_volatility_5d
        
        # TODO:
        # Calculate AC optimal trajectory
        # ... compute q_t using sinh formula ...
        
        # Convert to fraction adjustment like TWAP
        if volume > 0 and ehv_pct > 0:
            rem_pct_of_day = time_remaining / MINUTES_IN_DAY
            ehv_remaining = ehv_pct * rem_pct_of_day * volume
            required_fraction = (q_t / ehv_remaining) - 1.0
        else:
            required_fraction = 0.0
            
        required_fraction = np.clip(required_fraction, -1.0, 1.0)
        action_idx = np.argmin(np.abs(self._action_values - required_fraction))
        
        return int(action_idx), None

class POVBaseline:
    """
    Participation of Volume (POV) baseline strategy.
    Per user spec: set target as EHV_PCT by using neutral action each minute
    so the resulting rate follows the environment's expected horizon volume
    (no additional adjustment).
    """
    
    def __init__(self, name="POV_Baseline"):
        self.name = name
        logger.info(f"Initialized {self.name} strategy")
    
    def predict(self, observation, deterministic=True):
        """
        Predict neutral action (fraction = 0.0).
        
        @param observation: Current observation (unused)
        @param deterministic: Whether to use deterministic prediction (unused)
        @return: Tuple of (action, None) to match stable-baselines3 interface
        """
        return 5, None

# Dictionary mapping strategy names to classes for easy access
BASELINE_STRATEGIES = {
    'Random': RandomBaseline,
    'ConstantRate': ConstantRateBaseline,
    'AlmgrenChriss': AlmgrenChrissBaseline,
    'VWAP': VWAPBaseline,
    'AdaptiveVWAP': AdaptiveVWAPBaseline,
    'TWAP': TWAPBaseline,
    'POV': POVBaseline
}


def get_baseline_strategy(strategy_name, **kwargs):
    """
    Factory function to create baseline strategy instances.
    
    @param strategy_name: Name of the strategy (key in BASELINE_STRATEGIES)
    @param kwargs: Additional arguments to pass to the strategy constructor
    @return: Instance of the requested baseline strategy
    """
    if strategy_name not in BASELINE_STRATEGIES:
        raise ValueError(f"Unknown strategy: {strategy_name}. Available: {list(BASELINE_STRATEGIES.keys())}")
    
    return BASELINE_STRATEGIES[strategy_name](**kwargs) 