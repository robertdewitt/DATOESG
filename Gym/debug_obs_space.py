#!/usr/bin/env python3
"""
Debug script to check observation space compatibility
"""

import numpy as np
from gymnasium import spaces

def check_observation_space():
    """Check if the observation space definition is correct"""
    
    print("=== DEBUGGING OBSERVATION SPACE ===")
    
    # Current observation space definition from the code
    obs_dim = 14
    obs_low = np.array([
        0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -1.0, 0.0, 0.0, 0.0, 0.0, 0.0, -1.0, 0.0
    ], dtype=np.float32)
    obs_high = np.array([
        1e10, 1e16, 600.0, 1e16, 1.0, 1.0, 1.0, 1e10, 1e16, 1e10, 1e10, 1e10, 1.0, 1e10
    ], dtype=np.float32)
    
    print(f"obs_dim: {obs_dim}")
    print(f"obs_low length: {len(obs_low)}")
    print(f"obs_high length: {len(obs_high)}")
    print(f"obs_low: {obs_low}")
    print(f"obs_high: {obs_high}")
    
    # Check if lengths match
    if len(obs_low) != obs_dim:
        print(f"❌ ERROR: obs_low length ({len(obs_low)}) != obs_dim ({obs_dim})")
    if len(obs_high) != obs_dim:
        print(f"❌ ERROR: obs_high length ({len(obs_high)}) != obs_dim ({obs_dim})")
    if len(obs_low) != len(obs_high):
        print(f"❌ ERROR: obs_low length ({len(obs_low)}) != obs_high length ({len(obs_high)})")
    
    # Try to create observation space
    try:
        observation_space = spaces.Box(low=obs_low, high=obs_high, shape=(obs_dim,), dtype=np.float32)
        print(f"✅ Observation space created successfully: {observation_space}")
        
        # Test with a sample observation
        sample_obs = np.random.random(obs_dim).astype(np.float32) * 100  # Random values
        print(f"Sample observation shape: {sample_obs.shape}")
        print(f"Sample observation in space: {sample_obs in observation_space}")
        
        return observation_space
        
    except Exception as e:
        print(f"❌ ERROR creating observation space: {e}")
        return None

def check_stack_dimensions():
    """Check the dimensions being stacked in _get_observation"""
    
    print("\n=== CHECKING STACK DIMENSIONS ===")
    
    # The features being stacked (from _get_observation):
    features = [
        "mid_price",
        "volume", 
        "time_remaining",
        "shares_remaining",
        "adv_pct",
        "ehv_pct", 
        "signal",
        "last_fill_price",
        "last_trade_size",
        "immediate_impact",
        "accumulated_impact",
        "arrival_price",
        "regime",
        "vol_lag1"
        # vol5d removed
    ]
    
    print(f"Number of features being stacked: {len(features)}")
    print("Features:")
    for i, feature in enumerate(features):
        print(f"  {i+1}. {feature}")
    
    if len(features) != 14:
        print(f"❌ ERROR: Expected 14 features, got {len(features)}")
    else:
        print("✅ Feature count matches expected 14 dimensions")

if __name__ == "__main__":
    check_observation_space()
    check_stack_dimensions() 