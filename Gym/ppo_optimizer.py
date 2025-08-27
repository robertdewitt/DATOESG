import torch.nn as nn
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from stable_baselines3.common.callbacks import BaseCallback
import torch.nn.functional as F
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import VecNormalize
import logging
import time
import os
import torch

class LrCheck(BaseCallback):
    def _on_rollout_end(self) -> bool:
        lr = self.model.policy.optimizer.param_groups[0]["lr"]
        self.logger.record("debug/optimizer_lr", float(lr))
        return True

class ExecFeaturesOptimized(BaseFeaturesExtractor):
    """
    Optimized feature extractor with better performance
 
    It is optimized for performance and memory usage.
    It uses a single layer normalization and fused operations where possible.
    It also initializes the weights for faster convergence.
    It is used in the create_optimized_ppo_model function.
    """
    def __init__(self, observation_space, features_dim=256):
        super().__init__(observation_space, features_dim)
        
        n = observation_space.shape[0]
        
        # Single layer normalization (reused)
        self.norm = nn.LayerNorm(n)
        
        # Fused operations where possible
        self.trunk = nn.Sequential(
            nn.Linear(n, 256, bias=False),  
            nn.ReLU(inplace=True),          
            nn.Linear(256, features_dim),
            nn.ReLU(inplace=True),
        )
        
        # Initialize weights for faster convergence
        for m in self.trunk.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
        
        self._features_dim = features_dim
    
    def forward(self, obs):
        # Apply normalization separately (can be optimized by compiler)
        x = self.norm(obs)
        return self.trunk(x)


def create_optimized_ppo_model(train_env, model_name, num_envs=None, feature_extractor=None, ppo_kwargs=None, device="cpu", ppo_gamma=0.999):
    """
    Create an optimized PPO model with performance-tuned hyperparameters.
    
    Key optimizations:
    1. Larger batch sizes for better GPU utilization
    2. Optimized n_steps for efficient data collection
    3. Reduced epochs to prevent overfitting
    4. Tuned learning rate schedule

    Args:
        train_env: The environment to train the model on.
        model_name: The name of the model.
        num_envs: The number of environments to train the model on.
        feature_extractor: The feature extractor to use.
        ppo_kwargs: The PPO kwargs to use.
        device: The device to use for training.
        ppo_gamma: The PPO gamma to use.

    Returns:
        model: The optimized PPO model.
        n_steps: The number of steps per rollout.
        batch_size: The batch size.
    """
    
    # Auto-detect optimal batch configuration
    if num_envs is None:
        num_envs = train_env.num_envs
    
    if feature_extractor is None:
        feature_extractor = ExecFeaturesOptimized
    
    # OPTIMIZATION 1: Optimal n_steps and batch_size
    # Rule: n_steps * num_envs should be divisible by batch_size
    # Larger batches = better vectorization
    n_steps = 2048  # Reduced from 8192*4 for faster updates
    
    # Ensure batch_size divides evenly into total samples
    total_samples = n_steps * num_envs
    
    # Use larger batch size for better CPU/GPU utilization
    if total_samples >= 65536:
        batch_size = 8192
    elif total_samples >= 32768:
        batch_size = 4096
    elif total_samples >= 16384:
        batch_size = 2048
    else:
        batch_size = min(1024, total_samples)
    
    # Ensure divisibility
    while total_samples % batch_size != 0:
        batch_size = batch_size // 2

    # OPTIMIZATION 2: Reduced epochs for faster training
    n_epochs = 3 # 3-5 is usually sufficient with large batches

    # model_name to include settings
    full_model_name = f"{model_name}_nsteps_{n_steps}_batchsize_{batch_size}_epochs_{n_epochs}"

    # OPTIMIZATION 3: Optimized policy network
    if ppo_kwargs is None:
        ppo_kwargs = dict(
            features_extractor_class=ExecFeaturesOptimized,
            features_extractor_kwargs=dict(features_dim=256),
            net_arch=dict(pi=[256], vf=[256]),  # Smaller networks train faster
            activation_fn=nn.ReLU,
            ortho_init=False,  # Faster initialization
            share_features_extractor=True,  # Share features between actor and critic
            normalize_images=False,  # We handle normalization in the extractor
        )
    
    # OPTIMIZATION 4: Learning rate schedule
    def make_lr_schedule(base_lr):
        def lr_schedule(progress_remaining: float) -> float:
            return base_lr * progress_remaining     
        return lr_schedule

    # OPTIMIZATION 5: PPO hyperparameters
    model = PPO(
        policy="MlpPolicy",
        env=train_env,
        device=device,  
        n_steps=n_steps,
        batch_size=2048,
        learning_rate=make_lr_schedule(3e-4),  
        n_epochs=n_epochs,
        clip_range=0.2,  # Slightly higher for faster learning
        clip_range_vf=0.2,  # No value function clipping (faster)
        gae_lambda=0.95,
        vf_coef=0.5,
        max_grad_norm=0.5,  # Slightly higher for stability
        gamma=ppo_gamma,  # Slightly lower for faster convergence
        verbose=1,
        tensorboard_log=f'./tensorboard_logs/{full_model_name}',
        policy_kwargs=ppo_kwargs,
        # Additional optimizations
        use_sde=False,  # Don't use state-dependent exploration (slower)
        sde_sample_freq=-1,
        target_kl=0.01, 
        ent_coef=0.02,
        stats_window_size=100,  # Smaller window for faster stats computation
    )
    
    logging.info(f"\nOptimized Training Configuration for {full_model_name}:")
    logging.info(f"  - Parallel environments: {num_envs}")
    logging.info(f"  - Steps per rollout: {n_steps}")
    logging.info(f"  - Total samples per rollout: {total_samples:,}")
    logging.info(f"  - Batch size: {batch_size:,}")
    logging.info(f"  - Mini-batches per epoch: {total_samples // batch_size}")
    logging.info(f"  - Epochs per update: {n_epochs}")
    logging.info(f"  - Approximate updates per second: {1000 / (total_samples / batch_size * n_epochs):.1f}")
    
    return model, n_steps, batch_size, full_model_name

def train_ppo_fast(train_env, model_name, num_train_steps, mp_vec=None, callbacks=None, 
                   feature_extractor=None, ppo_kwargs=None, device="cpu", ppo_gamma=0.999):
    """
    Optimized training function with all performance improvements.

    Args:
        train_env: The environment to train the model on.
        model_name: The name of the model.
        num_train_steps: The number of steps to train the model for.
        mp_vec: The vectorized environment to train the model on.
        callbacks: The callbacks to use for training.
        feature_extractor: The feature extractor to use.
        ppo_kwargs: The PPO kwargs to use.
        device: The device to use for training.
        ppo_gamma: The PPO gamma to use.

    Returns:
        model: The trained model.
        train_time: The time it took to train the model.

    """    
    print(f"\n{'='*60}")
    print(f"OPTIMIZED PPO TRAINING: {model_name}")
    print(f"{'='*60}")

    start_time = time.time()
    
    # Add callback to check optimizer lr
    #callbacks = [LrCheck()] + (callbacks or [])
    
    # Create optimized model
    model, n_steps, batch_size, full_model_name = create_optimized_ppo_model(train_env, model_name)
    print("optimizer lr at init:", model.policy.optimizer.param_groups[0]["lr"])

    
    # Calculate actual training steps
    actual_train_steps = num_train_steps
    
    print(f"\nTraining {full_model_name} for {actual_train_steps:,} timesteps...")
    
    # OPTIMIZATION 6: Use compiled mode if available (PyTorch 2.0+)
    if hasattr(torch, 'compile') and torch.__version__ >= '2.0.0' and device != "cpu":
        print("Compiling model with torch.compile() for faster execution...")
        model.policy = torch.compile(model.policy, mode='reduce-overhead')
    
    # OPTIMIZATION 7: Set torch threads for CPU training
    if model.device.type == 'cpu':
        # Use all available cores
        # Keep workers light; avoid CPU oversubscription with 256 envs
        os.environ.setdefault("OMP_NUM_THREADS", "1")
        os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
        os.environ.setdefault("MKL_NUM_THREADS", "1")
        os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")
        num_threads = min(os.cpu_count(), 16)
        os.environ.setdefault("TORCH_NUM_THREADS",str(num_threads))           
        os.environ.setdefault("TORCH_NUM_INTEROP_THREADS", "1")
        torch.set_num_threads(int(os.environ["TORCH_NUM_THREADS"]))
        print(f"Using {int(num_threads)} CPU threads for training")
    
    # Train the model
    if mp_vec is not None:
            mp_vec.park_model(model, full_model_name, learn=True, steps=actual_train_steps, 
                  save=True, callbacks=callbacks, normalized=True, env=train_env)   
    else:
        model.learn(total_timesteps=actual_train_steps, callback=callbacks)
    
    train_time = time.time() - start_time
    
    # Calculate performance metrics
    steps_per_second = actual_train_steps / train_time
    time_per_thousand = 1000 / steps_per_second
    
    logging.info(f"\n{'='*60}")
    logging.info(f"TRAINING COMPLETED: {model_name}")
    logging.info(f"{'='*60}")
    logging.info(f"  Total training time: {train_time:.2f} seconds")
    logging.info(f"  Total timesteps: {actual_train_steps:,}")
    logging.info(f"  Performance: {steps_per_second:.1f} steps/second")
    logging.info(f"  Time per 1000 steps: {time_per_thousand:.2f} seconds")
    logging.info(f"  Expected remaining time for 1M steps: {(1_000_000 - actual_train_steps) / steps_per_second / 60:.1f} minutes")
    
    return model, train_time