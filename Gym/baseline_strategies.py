"""
Baseline strategies for optimal execution comparison.
These strategies provide simple benchmarks to compare against trained RL models.
"""

import numpy as np
import random
import logging

logger = logging.getLogger(__name__)


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
    Volume-Weighted Average Price (VWAP) baseline strategy.
    Uses the default ADV percentage rate, which is essentially VWAP-like trading.
    This corresponds to action 5 (fraction = 0.0).
    """
    
    def __init__(self, name="VWAP_Baseline"):
        self.name = name
        logger.info(f"Initialized {self.name} strategy")
    
    def predict(self, observation, deterministic=True):
        """
        Predict VWAP-style action (default ADV rate).
        
        @param observation: Current observation (unused for pure VWAP)
        @param deterministic: Whether to use deterministic prediction (unused)
        @return: Tuple of (action, None) to match stable-baselines3 interface
        """
        # Action 5 corresponds to fraction 0.0, which gives:
        # trade_size = adv_pct * (1 + 0) * current_volumes = adv_pct * current_volumes
        # This is the baseline ADV rate (VWAP-like)
        return 5, None


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


# Dictionary mapping strategy names to classes for easy access
BASELINE_STRATEGIES = {
    'Random': RandomBaseline,
    'ConstantRate': ConstantRateBaseline,
    'VWAP': VWAPBaseline,
    'AdaptiveVWAP': AdaptiveVWAPBaseline
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