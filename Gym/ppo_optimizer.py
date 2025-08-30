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
import math


class LrCheck(BaseCallback):
    def __init__(self, verbose: int = 0):
        super().__init__(verbose)

    def _on_step(self) -> bool:
        # required abstract method; do nothing each env step
        return True

    def _on_rollout_end(self) -> bool:
        # logs once per rollout/update
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



class ExecCNNExtractor(BaseFeaturesExtractor):
    """
    CNN that accepts single-step vectors (C,), stacks (C,K) or (K,C).
    - Uses SiLU activations but initializes with ReLU gain (works on all torch versions).
    - Pools adaptively so T=1 is safe (no kernel=2 crash).
    """
    def __init__(self, observation_space, features_dim=256):
        super().__init__(observation_space, features_dim)
        self._features_dim = features_dim
        self._lazy_first = True  # rebind first conv in forward() with true C

        def block(c_in, c_out, k=5, s=1, p=None, d=1):
            if p is None:
                p = d * (k - 1) // 2
            seq = nn.Sequential(
                nn.Conv1d(c_in, c_out, kernel_size=k, stride=s, padding=p, dilation=d, bias=True),
                nn.SiLU(),
                nn.GroupNorm(1, c_out),
            )
            # init conv with ReLU gain (good proxy for SiLU)
            conv = seq[0]
            nn.init.kaiming_uniform_(conv.weight, nonlinearity="relu")
            if conv.bias is not None:
                fan_in = conv.weight.shape[1] * conv.kernel_size[0]
                bound = 1 / math.sqrt(fan_in)
                nn.init.uniform_(conv.bias, -bound, bound)
            return seq

        # placeholder in_channels; will be rebound lazily
        self.stage1 = block(8, 64, k=5)
        self.stage2 = block(64, 64, k=5)
        self.stage3 = block(64, 128, k=5, d=2)  # dilation widens temporal field

        self.head = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128, features_dim),
            nn.SiLU(),
        )
        # init head
        lin = self.head[1]
        nn.init.xavier_uniform_(lin.weight, gain=nn.init.calculate_gain("relu"))
        if lin.bias is not None:
            nn.init.zeros_(lin.bias)

    @staticmethod
    def _to_bct(x: torch.Tensor) -> torch.Tensor:
        # Accept [B,C], [B,C,T], [B,T,C] -> return [B,C,T]
        if x.dim() == 2:              # [B,C] -> [B,C,1]
            x = x.unsqueeze(-1)
        elif x.shape[1] > x.shape[-1]:  # likely [B, T, C]
            x = x.permute(0, 2, 1).contiguous()
        return x  # [B,C,T]

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        x = self._to_bct(obs)
        B, C, T = x.shape

        # rebind first conv in stage1 with true C
        if self._lazy_first:
            conv = self.stage1[0]
            if conv.in_channels != C:
                new0 = nn.Conv1d(C, conv.out_channels,
                                 kernel_size=conv.kernel_size[0],
                                 stride=conv.stride[0],
                                 padding=conv.padding[0],
                                 dilation=conv.dilation[0],
                                 bias=True)
                nn.init.kaiming_uniform_(new0.weight, nonlinearity="relu")
                if new0.bias is not None:
                    fan_in = new0.weight.shape[1] * new0.kernel_size[0]
                    bound = 1 / math.sqrt(fan_in)
                    nn.init.uniform_(new0.bias, -bound, bound)
                self.stage1[0] = new0
            self._lazy_first = False

        # Block 1
        x = self.stage1(x)
        # Adaptive pool: when T==1, this becomes identity (ks=1, stride=1)
        pool_ks = max(1, x.shape[-1] // 2)
        x = F.max_pool1d(x, kernel_size=pool_ks, stride=pool_ks)

        # Blocks 2–3
        x = self.stage2(x)
        x = self.stage3(x)

        # Global pooling (safe even if T==1)
        x = F.adaptive_avg_pool1d(x, 1)   # [B,128,1]
        x = self.head(x)                  # [B,features_dim]
        return x




def create_optimized_ppo_model(train_env,  model_name, model=None, num_envs=None, feature_extractor=ExecFeaturesOptimized, ppo_kwargs=None, device="cpu", ppo_gamma=0.999, n_steps=2048):
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
            features_extractor_class=feature_extractor,
            features_extractor_kwargs=dict(features_dim=256),
            net_arch=dict(pi=[256,256,128], vf=[256,256,128]),
            activation_fn=nn.Tanh,
            ortho_init=False,
            share_features_extractor=True,
            normalize_images=False,
        )
    else:
        ppo_kwargs = ppo_kwargs.copy()
        ppo_kwargs.setdefault("features_extractor_class", feature_extractor)
        ppo_kwargs.setdefault("features_extractor_kwargs", dict(features_dim=256))
        ppo_kwargs.setdefault("ortho_init", False)
        ppo_kwargs.setdefault("share_features_extractor", True)
        ppo_kwargs.setdefault("normalize_images", False)
    
    # OPTIMIZATION 4: Learning rate schedule
    def make_linear_schedule(base_lr):
        def lr_schedule(progress_remaining: float) -> float:
            return base_lr * progress_remaining
        return lr_schedule

    if model is None:

        # OPTIMIZATION 5: PPO hyperparameters
        model = PPO(
            policy="MlpPolicy",
            env=train_env,
            device=device,  
            n_steps=n_steps,
            batch_size=batch_size,
            learning_rate=make_linear_schedule(3e-4), 
            n_epochs=n_epochs,
            clip_range=make_linear_schedule(0.18),
            clip_range_vf=None,
            gae_lambda=0.95,
            vf_coef=0.55,
            max_grad_norm=0.5,
            gamma=ppo_gamma,
            verbose=1,
            tensorboard_log=f'./tensorboard_logs/{full_model_name}',
            policy_kwargs=ppo_kwargs,
            # Additional optimizations
            use_sde=False,  # Don't use state-dependent exploration (slower)
            sde_sample_freq=-1,
            target_kl=0.02, 
            ent_coef=0.006,
            stats_window_size=100,  # Smaller window for faster stats computation
        )
    

    batch_side = model.batch_size
    n_steps = model.n_steps
    n_epochs = model.n_epochs
    total_samples = n_steps * num_envs


    logging.info(f"\nOptimized Training Configuration for {full_model_name}:")
    logging.info(f"  - Parallel environments: {num_envs}")
    logging.info(f"  - Steps per rollout: {n_steps}")
    logging.info(f"  - Total samples per rollout: {total_samples:,}")
    logging.info(f"  - Batch size: {batch_size:,}")
    logging.info(f"  - Mini-batches per epoch: {total_samples // batch_size}")
    logging.info(f"  - Epochs per update: {n_epochs}")
    logging.info(f"  - Approximate updates per second: {1000 / (total_samples / batch_size * n_epochs):.1f}")
    
    return model, n_steps, batch_size, full_model_name

def train_ppo_fast(train_env, model_name, num_train_steps, model=None, mp_vec=None, callbacks=None, 
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
    callbacks = (callbacks or [])
    callbacks = [LrCheck()] + callbacks
    
    # Create optimized model
    model, n_steps, batch_size, full_model_name = create_optimized_ppo_model(
        train_env,
        model_name,
        num_envs=train_env.num_envs,
        model=model,
        feature_extractor=feature_extractor,
        ppo_kwargs=ppo_kwargs,
        device=device,
        ppo_gamma=ppo_gamma,
    )
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