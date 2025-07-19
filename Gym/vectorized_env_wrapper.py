"""
Wrapper to make VectorizedMultiOrderExecutionEnv compatible with stable-baselines3.
"""

import gymnasium as gym
from gymnasium import spaces
import numpy as np
from optimal_execution_env_vectorized import VectorizedMultiOrderExecutionEnv
from stable_baselines3.common.vec_env import VecEnv
import warnings


class VectorizedEnvWrapper(gym.Env):
    """
    Wrapper that makes VectorizedMultiOrderExecutionEnv appear as a single environment
    for compatibility with stable-baselines3 training.
    """
    
    def __init__(self, stock_df_list, orders_df, impact_coef, decay_rate, num_envs,
                 min_rate=0.0, max_rate=0.1, window_size=1, unfilled_penalty=1e6,
                 device=None, render_mode=None, seed=None):
        """
        Initialize the wrapper.
        @param stock_df_list: List of DataFrames containing stock data.
        @param orders_df: DataFrame containing order data.
        @param impact_coef: Coefficient for market impact.
        @param decay_rate: Decay rate for impact.
        @param num_envs: Number of parallel environments to create.
        @param min_rate: Minimum trading rate.
        @param max_rate: Maximum trading rate.
        @param window_size: Size of the observation window.
        @param unfilled_penalty: Penalty for unfilled orders.
        @param device: Device to run the environment on (e.g., 'cpu', 'cuda').
        @param render_mode: Rendering mode (e.g., 'human', 'rgb_array').
        @param seed: Random seed for reproducibility.

        """
        self.vec_env = VectorizedMultiOrderExecutionEnv(
            stock_df_list=stock_df_list,
            orders_df=orders_df,
            impact_coef=impact_coef,
            decay_rate=decay_rate,
            num_envs=num_envs,
            min_rate=min_rate,
            max_rate=max_rate,
            window_size=window_size,
            unfilled_penalty=unfilled_penalty,
            device=device,
            render_mode=render_mode,
            seed=seed
        )
        
        # Set spaces from the vectorized environment
        self.observation_space = self.vec_env.observation_space
        self.action_space = self.vec_env.action_space
        self.render_mode = render_mode
        
        # Track current environment index for single-env interface
        self.current_env_idx = 0
    
    @property
    def env(self):
        """Provide access to underlying vectorized environment for compatibility."""
        return self.vec_env
    
    @property 
    def num_envs(self):
        """Number of parallel environments."""
        return self.vec_env.num_envs
        
    def reset(self, seed=None, options=None):
        """
        Reset and return observation for single environment interface.
        @param seed: Random seed for reproducibility.
        @param options: Additional options for reset (not used here).
        """
        # Call parent reset to handle seeding properly in gymnasium
        super().reset(seed=seed)
        
        # Reset the vectorized environment with the seed
        obs = self.vec_env.reset(seed=seed)
        
        # Debug: Check observation shape
        if hasattr(self, '_debug_obs_shape'):
            print(f"VecEnv obs shape: {obs.shape}, extracting env {self.current_env_idx}")
        
        # Ensure we extract the correct environment's observation
        if len(obs.shape) == 2:  # [num_envs, obs_dim]
            single_obs = obs[self.current_env_idx]
        else:  # Already single environment observation
            single_obs = obs
            
        return single_obs, {}
    
    def step(self, action):
        """
        Step and return results for single environment interface.
        @param action: Action to take (should be compatible with action space).
        @return: Tuple of (observation, reward, done, truncated, info)
        """
        # Create action array for all environments (use same action)
        actions = np.full(self.vec_env.num_envs, action)
        
        # Step the vectorized environment
        self.vec_env.step_async(actions)
        obs, rewards, dones, infos = self.vec_env.step_wait()
        
        # Return results for first environment
        return (
            obs[self.current_env_idx],
            rewards[self.current_env_idx], 
            dones[self.current_env_idx],
            False,  # truncated
            infos[self.current_env_idx]
        )
    
    def render(self):
        """
        Render the environment.
        @return: Rendered output (depends on render_mode).
        """
        return self.vec_env.render()
    
    def close(self):
        """
        Close the environment.
        """
        self.vec_env.close()
    
    def seed(self, seed=None):
        """
        Set the seed.
        @param seed: Random seed for reproducibility.
        
        WARNING: This method is deprecated. Use env.reset(seed=seed) instead.
        """
        warnings.warn(
            "The 'seed' method is deprecated. Use 'env.reset(seed=seed)' instead.",
            DeprecationWarning,
            stacklevel=2
        )
        # For backward compatibility, still support the old method
        if hasattr(self.vec_env, '_set_seed'):
            self.vec_env._set_seed(seed)
            return [seed]
        elif hasattr(self.vec_env, 'seed'):
            return self.vec_env.seed(seed)
        else:
            return [seed]
    
    def execute_orders(self, model, num_episodes=10, fixed_order_indices=None, return_indices=False):
        """
        Execute orders using the underlying vectorized environment.
        @param model: The trained RL model to use for action selection.
        @param num_episodes: Number of orders to execute in this run.
        @param fixed_order_indices: Optional list of specific order indices to use for consistent comparison.
        @param return_indices: If True, return tuple (orders, indices). If False, return only orders for backward compatibility.
        @return: List of order information dictionaries, or tuple (orders, indices) if return_indices=True.
        """
        orders, indices = self.vec_env.execute_orders(
            model=model, 
            num_episodes=num_episodes, 
            fixed_order_indices=fixed_order_indices
        )
        
        if return_indices:
            return orders, indices
        return orders

    def execute_fixed_orders(self, model, order_indices, return_indices=False):
        """
        Execute specific orders by their indices to ensure consistent evaluation across models.
        
        @param model: The trained RL model to use for action selection.
        @param order_indices: List of specific order indices to execute.
        @param return_indices: If True, return the order indices used (for compatibility).
        @return: List of order information dictionaries or tuple (orders, indices) if return_indices=True.
        """
        return self.vec_env.execute_fixed_orders(model, order_indices, return_indices)


def create_vectorized_env(stock_df_list, orders_df, impact_coef=1, decay_rate=5, 
                         num_envs=32, min_rate=0.0, max_rate=0.1, window_size=1, 
                         unfilled_penalty=1e6, device=None, render_mode=None, seed=None):
    """
    Factory function to create a vectorized environment.
    @param stock_df_list: List of DataFrames containing stock data.
    @param orders_df: DataFrame containing order data.
    @param impact_coef: Coefficient for market impact.
    @param decay_rate: Decay rate for impact.
    @param num_envs: Number of parallel environments to create.
    @param min_rate: Minimum trading rate.
    @param max_rate: Maximum trading rate.
    @param window_size: Size of the observation window.
    @param unfilled_penalty: Penalty for unfilled orders.
    @param device: Device to run the environment on (e.g., 'cpu', 'cuda').
    @param render_mode: Rendering mode (e.g., 'human', 'rgb_array').
    """
    return VectorizedEnvWrapper(
        stock_df_list=stock_df_list,
        orders_df=orders_df,
        impact_coef=impact_coef,
        decay_rate=decay_rate,
        num_envs=num_envs,
        min_rate=min_rate,
        max_rate=max_rate,
        window_size=window_size,
        unfilled_penalty=unfilled_penalty,
        device=device,
        render_mode=render_mode,
        seed=seed
    ) 

def compare_models_consistently(models_dict, env, num_episodes=10, order_indices=None, seed=None):
    """
    Compare multiple models on the same set of orders for fair evaluation.
    
    @param models_dict: Dictionary with model_name -> model pairs to compare.
    @param env: The execution environment (VectorizedEnvWrapper or VectorizedMultiOrderExecutionEnv).
    @param num_episodes: Number of orders to execute per model.
    @param order_indices: Optional specific order indices to use. If None, random orders will be generated.
    @param seed: Optional seed for reproducible order selection (only used if order_indices is None).
    @return: Dictionary with model_name -> (orders, indices) pairs, and 'common_indices' key.
    """
    # Note: seeding is handled when resetting the environment for execution
    
    results = {}
    common_order_indices = None
    
    # Execute first model to establish order indices
    first_model_name = list(models_dict.keys())[0]
    first_model = models_dict[first_model_name]
    
    print(f"Executing orders with {first_model_name} to establish order set...")
    orders, indices = env.execute_orders(
        model=first_model, 
        num_episodes=num_episodes, 
        fixed_order_indices=order_indices,
        return_indices=True
    )
    
    results[first_model_name] = orders
    common_order_indices = indices
    
    # Execute remaining models with the same order indices
    for model_name, model in list(models_dict.items())[1:]:
        print(f"Executing orders with {model_name} using same order set...")
        orders = env.execute_fixed_orders(model, common_order_indices)
        results[model_name] = orders
    
    # Add the common indices to results for reference
    results['common_indices'] = common_order_indices
    
    print(f"Successfully executed {len(common_order_indices)} orders across {len(models_dict)} models.")
    print(f"Order indices used: {common_order_indices}")
    
    return results 