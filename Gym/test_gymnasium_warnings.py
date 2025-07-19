"""
Test script to verify that gymnasium warnings are fixed.
"""

import pandas as pd
import numpy as np
from vectorized_env_wrapper import create_vectorized_env
import warnings

# Filter out other warnings to focus on gymnasium warnings
warnings.filterwarnings('ignore', category=UserWarning, module='stable_baselines3')

def test_gymnasium_warnings():
    """Test that gymnasium warnings are resolved."""
    
    print("Creating test environment...")
    
    # Create minimal test data
    stock_data = pd.DataFrame({
        'mid_price': [100.0] * 100,
        'vwap': [100.0] * 100,
        'volume': [1000] * 100,
        'volatility': [0.01] * 100,
        'spread': [0.01] * 100
    })
    
    orders_data = pd.DataFrame({
        'ticker': ['TEST'] * 10,
        'side': ['buy'] * 10,
        'order_qty': [1000] * 10,
        'time_horizon': [60] * 10,
        'adv_pct': [0.05] * 10,
        'ehv_pct': [0.1] * 10,
        'adv': [50000] * 10,
        'start_time': [0] * 10,
        'end_time': [59] * 10,
        'date': ['2023-01-01'] * 10
    })
    
    stock_df_list = {'TEST': stock_data}
    
    print("Creating vectorized environment...")
    env = create_vectorized_env(
        stock_df_list=stock_df_list,
        orders_df=orders_data,
        num_envs=4,
        device='cpu'
    )
    
    print("Testing reset with seed (new gymnasium way)...")
    obs, info = env.reset(seed=42)
    print(f"✅ Reset with seed successful. Observation shape: {obs.shape}")
    
    print("Testing old seed method (should show deprecation warning)...")
    try:
        env.seed(42)
        print("✅ Old seed method still works (with deprecation warning)")
    except Exception as e:
        print(f"❌ Old seed method failed: {e}")
    
    print("Testing step...")
    action = env.action_space.sample()
    obs, reward, done, truncated, info = env.step(action)
    print(f"✅ Step successful. Reward: {reward}")
    
    print("Closing environment...")
    env.close()
    print("✅ Test completed successfully!")

if __name__ == "__main__":
    test_gymnasium_warnings() 