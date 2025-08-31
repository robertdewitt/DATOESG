
import logging, os, json, math, pickle, random
from dataclasses import dataclass
from typing import List, Tuple, Optional

import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
from stable_baselines3.common.vec_env import VecNormalize

"""
Map Elites implementation for the optimal execution environment
"""


"""
Utility functions for map elites
"""

def niche_bins(B_LIQ, B_VOL):
    """
    Utility function to create a dictionary of niche bins.

    Args:
        B_LIQ: Number of liquidity bins
        B_VOL: Number of volatility bins
    """
    return {
        "hiL_loV": (int(0.85*B_LIQ), int(0.15*B_VOL)),
        "mid_mid": (int(0.50*B_LIQ), int(0.50*B_VOL)),
        "loL_hiV": (int(0.15*B_LIQ), int(0.85*B_VOL)),
    }


def select_specialist_for_order(archive_lv, order_liq, order_vol, 
                               liq_edges, vol_edges):
    """Select the best specialist for a given order's characteristics using quantile bins"""
    # Use the quantile binning function from your MAP-Elites code
    liq_bin = value_to_quantile_bin(order_liq, liq_edges)
    vol_bin = value_to_quantile_bin(order_vol, vol_edges)
    
    # No need to clip since value_to_quantile_bin already handles bounds
    
    # Get the specialist from that cell
    cell = archive_lv.grid[liq_bin][vol_bin]
    if cell:
        return cell
    else:
        # Fallback to best overall
        return archive_lv.best()


def park_cell_elite(archive, tag, ppo_model, mp, model_name_prefix, vecnorm=None):
    """
    Utility function to park a cell of the archive.

    Args:
        archive: The archive
        tag: The tag of the cell
        ppo_model: The PPO model
        mp: The model parking
        model_name_prefix: The prefix of the model name
        vecnorm: The VecNormalize object

    Returns:
        The name of the parked model
    """
    cell = archive.grid[tag[0]][tag[1]]
    if not cell: return None
    snap, fit, (liq, vol) = cell["snapshot"], cell["fitness"], cell["desc"]
    set_policy(ppo_model, snap, device="cpu")
    name = f"{model_name_prefix}__ME_{tag[2]}_f{fit:.6f}_liq{liq:.2f}_vol{vol:.2f}"
    mp.park_model(
        ppo_model,
        name,
        save=True,
        normalized=(vecnorm is not None),
        env=vecnorm
    )
    return name

import numpy as np
import matplotlib.pyplot as plt

def create_quantile_bins(data, n_bins):
    """
    Create quantile-based bin edges for data.
    
    Args:
        data: 1D array of values
        n_bins: Number of bins to create
    
    Returns:
        bin_edges: Array of bin edges (length n_bins + 1)
    """
    # Remove NaN values for quantile calculation
    clean_data = data[~np.isnan(data)]
    
    # Create quantiles from 0 to 1 with n_bins+1 points
    quantiles = np.linspace(0, 1, n_bins + 1)
    bin_edges = np.quantile(clean_data, quantiles)
    
    # Ensure edges are unique (handle case where many values are identical)
    bin_edges = np.unique(bin_edges)
    
    # If we lost bins due to duplicate values, pad with small increments
    while len(bin_edges) < n_bins + 1:
        # Add tiny increments to the last edge
        bin_edges = np.append(bin_edges, bin_edges[-1] + 1e-10)
    
    return bin_edges

def value_to_quantile_bin(value, bin_edges):
    """
    Convert a value to its quantile bin index.
    
    Args:
        value: Value to bin
        bin_edges: Array of bin edges from create_quantile_bins
    
    Returns:
        bin_index: Integer bin index
    """
    if np.isnan(value):
        return 0
    
    # Use searchsorted to find the appropriate bin
    bin_idx = np.searchsorted(bin_edges[1:], value, side='left')
    
    # Ensure we don't exceed the maximum bin index
    return min(bin_idx, len(bin_edges) - 2)

def create_lv_grid_quantile(B_LIQ, B_VOL, orders_view):
    """
    Create LV grid using quantile-based binning.
    
    Args:
        B_LIQ: Number of liquidity bins
        B_VOL: Number of volatility bins
        orders_view: Array with shape (n_samples, 2) where columns are [liquidity, volatility]
    
    Returns:
        cell2indices: Dictionary mapping (liq_bin, vol_bin) to list of indices
        nonempty_cells: List of non-empty (i,j) coordinates
        liq_bin_edges: Liquidity bin edges for reference
        vol_bin_edges: Volatility bin edges for reference
    """
    # Extract liquidity and volatility data
    liq_data = orders_view[:, 0]
    vol_data = orders_view[:, 1]
    
    # Create quantile-based bin edges
    liq_bin_edges = create_quantile_bins(liq_data, B_LIQ)
    vol_bin_edges = create_quantile_bins(vol_data, B_VOL)
    
    # Create the grid
    cell2indices = {}
    
    for idx, (liq, vol) in enumerate(orders_view):
        i = value_to_quantile_bin(liq, liq_bin_edges)
        j = value_to_quantile_bin(vol, vol_bin_edges)
        cell2indices.setdefault((i, j), []).append(idx)
    
    nonempty_cells = [ij for ij, L in cell2indices.items() if L]
    
    return cell2indices, nonempty_cells, liq_bin_edges, vol_bin_edges


def plot_lv_grid_density_quantile(lvgrid_to_indices, liq_bin_edges=None, vol_bin_edges=None):
    """
    Plot the liquidity and volatility population density with proper extent.
    """
    B_LIQ = max(i for (i, _j) in lvgrid_to_indices.keys()) + 1
    B_VOL = max(j for (_i, j) in lvgrid_to_indices.keys()) + 1
    
    density = np.zeros((B_LIQ, B_VOL), dtype=int)
    for (i, j), idxs in lvgrid_to_indices.items():
        density[i, j] = len(idxs)
    
    # Set extent so bin centers are at integer ticks
    extent = (-0.5, B_VOL - 0.5, -0.5, B_LIQ - 0.5)
    
    plt.figure(figsize=(12, 8))
    im = plt.imshow(density, cmap='viridis', origin='lower', aspect='auto', extent=extent)
    
    # Create colorbar with some padding from the main plot
    cbar = plt.colorbar(im, label='Population Density', pad=0.15)
    
    plt.xticks(np.arange(B_VOL))
    plt.yticks(np.arange(B_LIQ))
    plt.xlabel('Volatility bin')
    plt.ylabel('Liquidity bin')
    plt.title('Population Density in Liquidity-Volatility Grid (Quantile-Based)')
    
    # Add bin edge information if provided
    if liq_bin_edges is not None:
        # Add secondary y-axis with actual liquidity values
        ax2 = plt.gca().twinx()
        ax2.set_ylim(-0.5, B_LIQ - 0.5)
        ax2.set_yticks(np.arange(B_LIQ))
        ax2.set_yticklabels([f'{edge:.3f}' for edge in liq_bin_edges[:-1]], fontsize=8)
        # Fix the label positioning - use rotation=90 and increase labelpad
        ax2.set_ylabel('Liquidity Thresholds', rotation=90, labelpad=20)
    
    if vol_bin_edges is not None:
        # Add secondary x-axis with actual volatility values
        ax3 = plt.gca().twiny()
        ax3.set_xlim(-0.5, B_VOL - 0.5)
        ax3.set_xticks(np.arange(B_VOL))
        ax3.set_xticklabels([f'{edge:.4f}' for edge in vol_bin_edges[:-1]], fontsize=8, rotation=45)
        ax3.set_xlabel('Volatility Thresholds')
    
    plt.tight_layout()
    plt.show()

def _to_bin(x, B): 
    """
    Utility function to convert a value to a bin.

    Args:
        x: Value to convert
        B: Number of bins
    """
    try:
        x = float(x)
        if x <= 0: return 0
        if x >= 1: return B-1
        return int(x * B)
    except Exception:
        logging.error("Bad bin value: %r", x)
        return 0

def create_lv_grid(B_LIQ, B_VOL, orders_view):
    """
    Utility function to create a grid of liquidity and volatility bins.

    Args:
        B_LIQ: Number of liquidity bins
        B_VOL: Number of volatility bins
        orders_view: Orders view

    Returns:
        cell2indices: Dictionary of liquidity and volatility bins
    """
    cell2indices = {}
    average_liq = 0
    average_vol = 0
    for idx, (liq, vol) in enumerate(orders_view):
        average_liq += liq
        average_vol += vol
        i = _to_bin(liq, B_LIQ); j = _to_bin(vol, B_VOL)
        cell2indices.setdefault((i,j), []).append(idx)
    nonempty_cells = [ij for ij, L in cell2indices.items() if L]
    return cell2indices, nonempty_cells


"""
Analysis utility functions
"""
def plot_lv_archive(archive, title="MAP-Elites (liq–vol)"):
    H, W = archive.bL, archive.bV
    Z = np.full((H, W), np.nan, float)
    for i in range(H):
        for j in range(W):
            cell = archive.grid[i][j]
            if cell: Z[i, j] = cell["fitness"]
    plt.figure(figsize=(7,6))
    plt.imshow(Z.T, origin="lower", aspect="auto", interpolation="nearest")
    plt.colorbar(label="fitness (−mean cost)")
    plt.title(title); plt.xlabel("Liquidity bin"); plt.ylabel("Volatility bin")
    plt.show()



"""
Tiny, self-contained driver that:

seeds each non-empty (liq, vol) cell with the PPO seed (1 eval),

mutates around it and competes within the same cell,

keeps the best model per cell.

"""
class LiquidityVolatilityArchive:
    def __init__(self, bL, bV):
        self.bL, self.bV = bL, bV
        self.grid = [[None for _ in range(bV)] for _ in range(bL)]

    def insert(self, i, j, snapshot, fitness, desc):
        cell = self.grid[i][j]
        if (cell is None) or (fitness > cell["fitness"]):
            self.grid[i][j] = {"snapshot": snapshot, "fitness": float(fitness), "desc": tuple(desc)}

    def best(self):
        best = None
        for i in range(self.bL):
            for j in range(self.bV):
                cell = self.grid[i][j]
                if cell is None: 
                    continue
                if (best is None) or (cell["fitness"] > best["fitness"]):
                    best = {"i": i, "j": j, **cell}
        return best

# Vecnorm policy aware wrapper

class VecNormPolicyWrapper:
    """
    Wraps an SB3 PPO model; normalizes observations with saved VecNormalize stats
    before calling .predict(...).
    """
    def __init__(self, base_model, vecnorm):
        self.base = base_model          # SB3 PPO
        self.vecnorm = vecnorm          # loaded VecNormalize or None
        if vecnorm is not None:
            vecnorm.training = False
            vecnorm.norm_obs = True
            vecnorm.norm_reward = False

    @property
    def policy(self):
        return self.base.policy

    # for your driver: allow weight loading on the base PPO
    def load_policy_snapshot(self, snap, device="cpu"):
        snap_dev = {k: (v.to(device) if isinstance(v, torch.Tensor) else v) for k, v in snap.items()}
        self.base.policy.load_state_dict(snap_dev, strict=True)

    def _normalize_obs(self, obs):
        """
        Normalize the observations.

        Args:
            obs: The observations to normalize

        Returns:
            The normalized observations
        """
        if self.vecnorm is None:
            return obs
        # ensure numpy float32
        x = np.asarray(obs, dtype=np.float32)
        mean = self.vecnorm.obs_rms.mean
        var  = self.vecnorm.obs_rms.var
        eps  = float(getattr(self.vecnorm, "epsilon", 1e-8))
        clip = float(getattr(self.vecnorm, "clip_obs", 10.0))
        x = (x - mean) / np.sqrt(var + eps)
        return np.clip(x, -clip, clip)

    def predict(self, obs, deterministic=True):
        obs_norm = self._normalize_obs(obs)
        return self.base.predict(obs_norm, deterministic=deterministic)

def snapshot_policy(model):
    """
    Copy SB3 policy weights (no optimizer) to CPU tensors.
    
    Args:
        model: The model to snapshot

    Returns:
        A dictionary of the policy weights
    """
    return {k: v.detach().cpu().clone() for k, v in model.policy.state_dict().items()}

def set_policy(model, snap, device="cpu"):
    """
    Load a policy snapshot into the model.
    
    Args:
        model: The model to set the policy for
        snap: The policy snapshot to load
        device: The device to load the policy on
    """
    if device == "auto":
        device = next(model.policy.parameters()).device.type
    if device not in ["cpu", "cuda", "mps"]:
        raise ValueError(f"Invalid device: {device}")
    snap_dev = {k: (v.to(device) if isinstance(v, torch.Tensor) else v) for k, v in snap.items()}
    model.policy.load_state_dict(snap_dev, strict=True)

def mutate_policy_snapshot(snap, sigma=0.01):
    """
    Return a mutated copy of a policy snapshot (Gaussian noise for float tensors).
    
    Args:
        snap: The policy snapshot to mutate
        sigma: The standard deviation of the Gaussian noise

    Returns:
        A dictionary of the mutated policy weights
    """
    out = {}
    for k, v in snap.items():
        if isinstance(v, torch.Tensor) and v.is_floating_point():
            out[k] = v + sigma * torch.randn_like(v)
        else:
            out[k] = v.clone() if isinstance(v, torch.Tensor) else v
    return out




def _fitness_from_orders(orders):
    costs=[]
    for tr in orders:
        if not tr: continue
        step = [s.get("total_step_cost") for s in tr if "total_step_cost" in s]
        costs.append(sum(step) if step else tr[-1].get("total_cost", 0.0))
    return -float(np.mean(costs)) if costs else -np.inf



def map_elites_liquidity_volatility_seeded(
    env,
    ppo_seed,
    cell2indices,
    nonempty_cells,
    orders_df,
    B_LIQ=20, B_VOL=20,
    iters=60,
    children_per_iter=64,
    eval_k=4,
    sigma=0.01,
    device="cpu",
    vecnorm=None
):
    archive = LiquidityVolatilityArchive(B_LIQ, B_VOL)
    
    seed = VecNormPolicyWrapper(ppo_seed, vecnorm)
    seed.policy.eval()

    # take a baseline snapshot of the policy weights
    seed_snap = snapshot_policy(seed.base)

    def _sample_idxs(i, j, k):
        L = cell2indices.get((i, j), [])
        if not L: 
            return []
        if len(L) >= k:
            import random
            return random.sample(L, k)
        return [random.choice(L) for _ in range(k)]

    # --- Seeding: evaluate the PPO seed in each non-empty cell ---
    logging.info(f"Seeding: evaluating the PPO seed in each non-empty cell")
    for (i, j) in tqdm(nonempty_cells, desc="Seeding: evaluating the PPO seed in each non-empty cell"):
        idxs = _sample_idxs(i, j, eval_k)
        if not idxs:
            continue
        seed.load_policy_snapshot(seed_snap, device=device)
        orders = env.execute_orders_vectorized(
            seed, idxs, collect_step_info=True, show_progress=False,
            model_name=f"PPO-seed@{i}-{j}"
        )
        fit = _fitness_from_orders(orders)
        liq = float(orders_df.loc[idxs, "liq_norm"].mean())
        vol = float(orders_df.loc[idxs, "vol_norm"].mean())
        archive.insert(i, j, seed_snap, fit, (liq, vol))

    # --- Illumination loop ---
    logging.info(f"Illumination loop: iterating {iters} times")
    for _ in tqdm(range(iters), desc="Illumination loop: iterating over iterations",position=1, leave=False):
        targets = random.choices(nonempty_cells, k=min(children_per_iter, len(nonempty_cells)))
        for (i, j) in targets:
            parent_snap = archive.grid[i][j]["snapshot"] if archive.grid[i][j] else seed_snap
            child_snap = mutate_policy_snapshot(parent_snap, sigma=sigma)

            idxs = _sample_idxs(i, j, eval_k)
            if not idxs:
                continue
            seed.load_policy_snapshot(child_snap, device=device)
            orders = env.execute_orders_vectorized(
                seed, idxs, collect_step_info=True, show_progress=False,
                model_name=f"child@{i}-{j}"
            )
            fit = _fitness_from_orders(orders)
            liq = float(orders_df.loc[idxs, "liq_norm"].mean())
            vol = float(orders_df.loc[idxs, "vol_norm"].mean())
            archive.insert(i, j, child_snap, fit, (liq, vol))

    # restore seed weights just to be tidy
    seed.load_policy_snapshot(seed_snap, device=device)
    return archive



# Enhanced MAP-Elites with progress tracking
def map_elites_with_tracking(env, ppo_seed, cell2indices, nonempty_cells, orders_df,
                             B_LIQ, B_VOL, iters, children_per_iter, eval_k, sigma,
                             device="cpu", vecnorm=None):
    """Modified MAP-Elites that tracks archive state at each iteration"""
    
    archive = LiquidityVolatilityArchive(B_LIQ, B_VOL)
    archive_snapshots = []  # Store archive state over time
    
    seed = VecNormPolicyWrapper(ppo_seed, vecnorm)
    seed.policy.eval()
    seed_snap = snapshot_policy(seed.base)
    
    def _sample_idxs(i, j, k):
        L = cell2indices.get((i, j), [])
        if not L: return []
        if len(L) >= k:
            return random.sample(L, k)
        return [random.choice(L) for _ in range(k)]
    
    def capture_archive_state():
        """Capture current state of archive for visualization"""
        state = np.full((B_LIQ, B_VOL), np.nan)
        coverage = 0
        total_fitness = 0
        for i in range(B_LIQ):
            for j in range(B_VOL):
                cell = archive.grid[i][j]
                if cell:
                    state[i, j] = cell["fitness"]
                    coverage += 1
                    total_fitness += cell["fitness"]
        return {
            'grid': state.copy(),
            'coverage': coverage,
            'avg_fitness': total_fitness / coverage if coverage > 0 else 0,
            'best_fitness': archive.best()["fitness"] if archive.best() else -np.inf
        }
    
    # Seeding phase
    print(f"  Seeding {len(nonempty_cells)} cells...")
    for (i, j) in tqdm(nonempty_cells, desc="  Seeding", leave=False):
        idxs = _sample_idxs(i, j, eval_k)
        if not idxs: continue
        
        seed.load_policy_snapshot(seed_snap, device=device)
        orders = env.execute_orders_vectorized(
            seed, idxs, collect_step_info=True, show_progress=False,
            model_name=f"PPO-seed@{i}-{j}"
        )
        fit = _fitness_from_orders(orders)
        liq = float(orders_df.loc[idxs, "liq_norm"].mean())
        vol = float(orders_df.loc[idxs, "vol_norm"].mean())
        archive.insert(i, j, seed_snap, fit, (liq, vol))
    
    # Capture initial state after seeding
    archive_snapshots.append(capture_archive_state())
    
    # Illumination loop with progress tracking
    print(f"  Illumination: {iters} iterations × {children_per_iter} children")
    for iter_idx in tqdm(range(iters), desc="  Illumination", leave=False):
        targets = random.choices(nonempty_cells, k=min(children_per_iter, len(nonempty_cells)))
        
        improvements = 0
        for (i, j) in targets:
            parent_snap = archive.grid[i][j]["snapshot"] if archive.grid[i][j] else seed_snap
            child_snap = mutate_policy_snapshot(parent_snap, sigma=sigma)
            
            idxs = _sample_idxs(i, j, eval_k)
            if not idxs: continue
            
            old_fitness = archive.grid[i][j]["fitness"] if archive.grid[i][j] else -np.inf
            
            seed.load_policy_snapshot(child_snap, device=device)
            orders = env.execute_orders_vectorized(
                seed, idxs, collect_step_info=True, show_progress=False,
                model_name=f"child@{i}-{j}"
            )
            fit = _fitness_from_orders(orders)
            
            if fit > old_fitness:
                improvements += 1
                liq = float(orders_df.loc[idxs, "liq_norm"].mean())
                vol = float(orders_df.loc[idxs, "vol_norm"].mean())
                archive.insert(i, j, child_snap, fit, (liq, vol))
        
        # Track progress every 10 iterations
        if (iter_idx + 1) % 10 == 0:
            archive_snapshots.append(capture_archive_state())
            if (iter_idx + 1) % 20 == 0:
                state = archive_snapshots[-1]
                print(f"    Iter {iter_idx+1}: Coverage={state['coverage']}/{B_LIQ*B_VOL}, "
                      f"Best={state['best_fitness']:.4f}, Avg={state['avg_fitness']:.4f}, "
                      f"Improvements={improvements}/{len(targets)}")
    
    # Capture final state
    archive_snapshots.append(capture_archive_state())
    
    seed.load_policy_snapshot(seed_snap, device=device)
    return archive, archive_snapshots


def map_elites_with_capture(env, ppo_seed, cell2indices, nonempty_cells, orders_df,
                            B_LIQ, B_VOL, iters=100, children_per_iter=256, eval_k=4, 
                            sigma=0.015, capture_every=5, vecnorm=None):
    """MAP-Elites that captures grid states for animation"""
    archive = LiquidityVolatilityArchive(B_LIQ, B_VOL)
    grid_history = []
    
    # Force CPU for PPO models
    if hasattr(ppo_seed, 'device'):
        ppo_seed.device = 'cpu'
    
    seed = VecNormPolicyWrapper(ppo_seed, vecnorm)
    seed.policy.eval()
    seed_snap = snapshot_policy(seed.base)
    
    def _sample_idxs(i, j, k):
        L = cell2indices.get((i, j), [])
        if not L: return []
        return random.sample(L, k) if len(L) >= k else [random.choice(L) for _ in range(k)]
    
    def capture_grid():
        Z = np.full((B_LIQ, B_VOL), np.nan)
        for i in range(B_LIQ):
            for j in range(B_VOL):
                if archive.grid[i][j]:
                    Z[i,j] = archive.grid[i][j]["fitness"]
        return Z.copy()
    
    # Seeding with progress bar
    for (i, j) in tqdm(nonempty_cells, desc="  Seeding cells"):
        idxs = _sample_idxs(i, j, eval_k)
        if not idxs: continue
        seed.load_policy_snapshot(seed_snap, device="cpu")
        orders = env.execute_orders_vectorized(
            seed, idxs, collect_step_info=True, show_progress=False
        )
        fit = _fitness_from_orders(orders)
        liq = float(orders_df.loc[idxs, "liq_norm"].mean())
        vol = float(orders_df.loc[idxs, "vol_norm"].mean())
        archive.insert(i, j, seed_snap, fit, (liq, vol))
    
    grid_history.append(capture_grid())
    
    # Illumination with progress bar
    for iter_idx in tqdm(range(iters), desc="  MAP-Elites iterations"):
        targets = random.choices(nonempty_cells, k=min(children_per_iter, len(nonempty_cells)))
        for (i, j) in targets:
            parent_snap = archive.grid[i][j]["snapshot"] if archive.grid[i][j] else seed_snap
            child_snap = mutate_policy_snapshot(parent_snap, sigma=sigma)
            
            idxs = _sample_idxs(i, j, eval_k)
            if not idxs: continue
            
            seed.load_policy_snapshot(child_snap, device="cpu")
            orders = env.execute_orders_vectorized(
                seed, idxs, collect_step_info=True, show_progress=False
            )
            fit = _fitness_from_orders(orders)
            liq = float(orders_df.loc[idxs, "liq_norm"].mean())
            vol = float(orders_df.loc[idxs, "vol_norm"].mean())
            archive.insert(i, j, child_snap, fit, (liq, vol))
        
        if (iter_idx + 1) % capture_every == 0:
            grid_history.append(capture_grid())
    
    seed.load_policy_snapshot(seed_snap, device="cpu")
    return archive, grid_history