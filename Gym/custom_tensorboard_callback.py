"""
Custom TensorBoard callback for logging additional training statistics.
"""

import numpy as np
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.logger import Logger
import torch


class CustomTensorBoardCallback(BaseCallback):
    """
    Custom callback to log additional training statistics to TensorBoard.
    """
    
    def __init__(self, verbose: int = 0):
        super().__init__(verbose)
        self.episode_rewards = []
        self.episode_lengths = []
        self.action_distributions = []
        self.value_predictions = []
        self.policy_losses = []
        self.value_losses = []
        
    def _on_step(self) -> bool:
        """
        Called after each step during training.
        """
        # Log episode statistics if episode is done
        if self.locals.get('dones', [False])[0]:
            # Episode reward
            episode_reward = self.locals.get('rewards', [0])[0]
            self.episode_rewards.append(episode_reward)
            
            # Episode length (you might need to track this separately)
            # For now, we'll use a simple counter
            
            # Log to TensorBoard
            self.logger.record("custom/episode_reward", episode_reward)
            self.logger.record("custom/mean_episode_reward", np.mean(self.episode_rewards[-100:]))
            self.logger.record("custom/std_episode_reward", np.std(self.episode_rewards[-100:]))
            
        # Log action distribution
        if 'actions' in self.locals:
            actions = self.locals['actions']
            if isinstance(actions, np.ndarray):
                action_counts = np.bincount(actions.flatten(), minlength=11)  # 11 actions
                action_probs = action_counts / np.sum(action_counts)
                for i, prob in enumerate(action_probs):
                    self.logger.record(f"custom/action_{i}_prob", prob)
        
        # Log value predictions
        if 'values' in self.locals:
            values = self.locals['values']
            if isinstance(values, np.ndarray):
                self.logger.record("custom/mean_value", np.mean(values))
                self.logger.record("custom/std_value", np.std(values))
                self.logger.record("custom/min_value", np.min(values))
                self.logger.record("custom/max_value", np.max(values))
        
        # Log policy entropy
        if 'entropy' in self.locals:
            entropy = self.locals['entropy']
            self.logger.record("custom/policy_entropy", entropy)
        
        # Log learning rate
        if hasattr(self.model, 'learning_rate'):
            self.logger.record("custom/learning_rate", self.model.learning_rate)
        
        return True
    
    def _on_rollout_end(self) -> None:
        """
        Called at the end of each rollout.
        """
        # Log rollout statistics
        if hasattr(self.model, 'rollout_buffer'):
            rollout_buffer = self.model.rollout_buffer
            
            # Log rollout buffer statistics
            if hasattr(rollout_buffer, 'observations'):
                obs = rollout_buffer.observations
                if isinstance(obs, np.ndarray):
                    self.logger.record("custom/obs_mean", np.mean(obs))
                    self.logger.record("custom/obs_std", np.std(obs))
                    self.logger.record("custom/obs_min", np.min(obs))
                    self.logger.record("custom/obs_max", np.max(obs))
            
            if hasattr(rollout_buffer, 'rewards'):
                rewards = rollout_buffer.rewards
                if isinstance(rewards, np.ndarray):
                    self.logger.record("custom/rollout_reward_mean", np.mean(rewards))
                    self.logger.record("custom/rollout_reward_std", np.std(rewards))
                    self.logger.record("custom/rollout_reward_min", np.min(rewards))
                    self.logger.record("custom/rollout_reward_max", np.max(rewards))
            
            if hasattr(rollout_buffer, 'values'):
                values = rollout_buffer.values
                if isinstance(values, np.ndarray):
                    self.logger.record("custom/rollout_value_mean", np.mean(values))
                    self.logger.record("custom/rollout_value_std", np.std(values))
        
        # Log model parameters statistics
        if hasattr(self.model, 'policy'):
            policy = self.model.policy
            
            # Log policy network weights statistics
            for name, param in policy.named_parameters():
                if param.requires_grad:
                    param_np = param.detach().cpu().numpy()
                    self.logger.record(f"custom/policy_{name}_mean", np.mean(param_np))
                    self.logger.record(f"custom/policy_{name}_std", np.std(param_np))
                    self.logger.record(f"custom/policy_{name}_norm", np.linalg.norm(param_np))
            
            # Log value function statistics
            if hasattr(policy, 'value_net'):
                value_params = list(policy.value_net.parameters())
                if value_params:
                    value_norm = sum(p.norm().item() for p in value_params)
                    self.logger.record("custom/value_net_norm", value_norm)
        
        # Log gradient statistics
        if hasattr(self.model, 'policy'):
            policy = self.model.policy
            total_norm = 0
            param_count = 0
            
            for p in policy.parameters():
                if p.grad is not None:
                    param_norm = p.grad.data.norm(2)
                    total_norm += param_norm.item() ** 2
                    param_count += 1
            
            if param_count > 0:
                total_norm = total_norm ** (1. / 2)
                self.logger.record("custom/gradient_norm", total_norm)
                self.logger.record("custom/gradient_norm_per_param", total_norm / param_count)


class EnvironmentStatsCallback(BaseCallback):
    """
    Callback to log environment-specific statistics.
    """
    
    def __init__(self, verbose: int = 0):
        super().__init__(verbose)
        
    def _on_step(self) -> bool:
        """
        Log environment-specific statistics.
        """
        # Get environment info if available
        if 'infos' in self.locals:
            infos = self.locals['infos']
            if isinstance(infos, list) and len(infos) > 0:
                info = infos[0]  # First environment
                
                # Log order execution statistics
                if 'shares_remaining' in info:
                    self.logger.record("env/shares_remaining", info['shares_remaining'])
                
                if 'order_vwap' in info:
                    self.logger.record("env/order_vwap", info['order_vwap'])
                
                if 'arrival_price' in info:
                    self.logger.record("env/arrival_price", info['arrival_price'])
                
                if 'immediate_impact' in info:
                    self.logger.record("env/immediate_impact", info['immediate_impact'])
                
                if 'accumulated_impact' in info:
                    self.logger.record("env/accumulated_impact", info['accumulated_impact'])
                
                # Calculate slippage if possible
                if 'order_vwap' in info and 'arrival_price' in info:
                    if info['arrival_price'] > 0:
                        slippage = (info['arrival_price'] - info['order_vwap']) / info['arrival_price']
                        self.logger.record("env/slippage_bps", slippage * 10000)  # Convert to basis points
                
                # Log action statistics
                if 'action_percentage' in info:
                    self.logger.record("env/action_percentage", info['action_percentage'])
                
                # Log completion statistics
                if 'shares_remaining' in info and 'order_qty' in info:
                    completion_ratio = 1 - (info['shares_remaining'] / info['order_qty'])
                    self.logger.record("env/completion_ratio", completion_ratio)
        
        return True


class TrainingDebugCallback(BaseCallback):
    """
    Callback to log debugging information during training.
    """
    
    def __init__(self, verbose: int = 0):
        super().__init__(verbose)
        self.step_count = 0
        
    def _on_step(self) -> bool:
        """
        Log debugging information.
        """
        self.step_count += 1
        
        # Log basic training info
        self.logger.record("debug/step_count", self.step_count)
        
        # Log memory usage (if available)
        try:
            import psutil
            process = psutil.Process()
            memory_info = process.memory_info()
            self.logger.record("debug/memory_mb", memory_info.rss / 1024 / 1024)
        except ImportError:
            pass
        
        # Log GPU memory if using CUDA
        if torch.cuda.is_available():
            self.logger.record("debug/gpu_memory_mb", torch.cuda.memory_allocated() / 1024 / 1024)
            self.logger.record("debug/gpu_memory_reserved_mb", torch.cuda.memory_reserved() / 1024 / 1024)
        
        # Log observation statistics
        if 'observations' in self.locals:
            obs = self.locals['observations']
            if isinstance(obs, np.ndarray):
                self.logger.record("debug/obs_shape", obs.shape)
                self.logger.record("debug/obs_mean", np.mean(obs))
                self.logger.record("debug/obs_std", np.std(obs))
                
                # Log individual observation components
                if obs.shape[-1] >= 15:  # Assuming 15 observation dimensions
                    obs_names = [
                        'mid_price', 'volume', 'time_remaining', 'shares_remaining', 'adv_pct',
                        'ehv_pct', 'signal', 'last_fill_price', 'last_trade_size', 'immediate_impact',
                        'accumulated_impact', 'arrival_price', 'regime', 'daily_vol_lag1', 'daily_vol_5d'
                    ]
                    
                    for i, name in enumerate(obs_names):
                        if i < obs.shape[-1]:
                            self.logger.record(f"debug/obs_{name}_mean", np.mean(obs[..., i]))
                            self.logger.record(f"debug/obs_{name}_std", np.std(obs[..., i]))
        
        return True 