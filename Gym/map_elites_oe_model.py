
import logging, os, json, math, pickle, random
from dataclasses import dataclass
from typing import List, Tuple, Optional

import numpy as np
import pandas as pd
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


def plot_archive_surfaces(train_grid, test_grid, baseline_train, baseline_test,
                          B_LIQ=3, B_VOL=3, title="MAP-Elites Archive",
                          save_path=None, ensemble_test_grid=None):
    """
    Side-by-side 3D surface plots.
    Specialist surface coloured green/red vs baseline;
    baseline shown as semi-transparent red plane.
    If ensemble_test_grid is provided a third panel is added.
    """
    panels = [
        (train_grid, baseline_train, "Train (Specialist)"),
        (test_grid, baseline_test, "Test (Specialist)"),
    ]
    if ensemble_test_grid is not None:
        panels.append(
            (ensemble_test_grid, baseline_test, "Test (Ensemble)")
        )

    n_panels = len(panels)
    fig = plt.figure(figsize=(8 * n_panels, 7))
    labels = (
        ["Low", "Med", "High"] if B_LIQ == 3
        else [str(k) for k in range(B_LIQ)]
    )
    x = np.arange(B_LIQ)
    y = np.arange(B_VOL)
    X, Y = np.meshgrid(x, y, indexing='ij')

    for panel, (grid, bl, ptitle) in enumerate(panels):
        ax = fig.add_subplot(1, n_panels, panel + 1, projection='3d')
        Z = np.ma.masked_invalid(grid.copy())

        fc = np.empty((B_LIQ - 1, B_VOL - 1, 4))
        for i in range(B_LIQ - 1):
            for j in range(B_VOL - 1):
                avg = np.nanmean(
                    [grid[i, j], grid[i + 1, j],
                     grid[i, j + 1], grid[i + 1, j + 1]]
                )
                fc[i, j] = (
                    (0.18, 0.80, 0.44, 0.85) if avg >= bl
                    else (0.91, 0.30, 0.24, 0.85)
                )

        ax.plot_surface(
            X, Y, Z, facecolors=fc,
            edgecolor='k', linewidth=0.6, shade=True,
        )

        xx, yy = np.meshgrid(
            np.linspace(-0.5, B_LIQ - 0.5, 10),
            np.linspace(-0.5, B_VOL - 0.5, 10),
        )
        ax.plot_surface(
            xx, yy, np.full_like(xx, bl), alpha=0.25, color='red',
        )

        ax.set_xlabel('Liquidity')
        ax.set_ylabel('Volatility')
        ax.set_zlabel('Fitness (-cost)')
        ax.set_title(ptitle)
        ax.set_xticks(range(B_LIQ))
        ax.set_xticklabels(labels, fontsize=8)
        ax.set_yticks(range(B_VOL))
        ax.set_yticklabels(labels, fontsize=8)

    fig.suptitle(title, fontsize=14, fontweight='bold')
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved: {save_path}")
    plt.show()


def plot_archive_heatmap(train_grid, test_grid,
                         baseline_train, baseline_test,
                         B_LIQ=3, B_VOL=3,
                         title="MAP-Elites Archive",
                         save_path=None,
                         ensemble_test_grid=None):
    """
    Side-by-side annotated heatmaps.
    Colour = specialist delta vs baseline (RdYlGn diverging).
    Each cell annotated with fitness value and percentage delta.
    If ensemble_test_grid is provided a third panel is added.
    """
    from matplotlib.colors import TwoSlopeNorm

    panels = [
        (train_grid, baseline_train, "Train (Specialist)"),
        (test_grid, baseline_test, "Test (Specialist)"),
    ]
    if ensemble_test_grid is not None:
        panels.append(
            (ensemble_test_grid, baseline_test, "Test (Ensemble)")
        )

    n_panels = len(panels)
    fig, axes = plt.subplots(
        1, n_panels, figsize=(7 * n_panels, 5.5),
    )
    if n_panels == 1:
        axes = [axes]
    tick_labels = (
        ["Low", "Med", "High"] if B_LIQ == 3
        else [str(k) for k in range(B_LIQ)]
    )

    for ax, (grid, bl, ptitle) in zip(axes, panels):
        diff = grid - bl
        vmax = np.nanmax(np.abs(diff))
        if vmax == 0:
            vmax = 1e-6
        norm = TwoSlopeNorm(vmin=-vmax, vcenter=0, vmax=vmax)

        im = ax.imshow(
            diff.T, origin='lower', cmap='RdYlGn',
            norm=norm, aspect='equal',
        )

        for i in range(B_LIQ):
            for j in range(B_VOL):
                val = grid[i, j]
                if np.isnan(val):
                    ax.text(
                        i, j, "\u2014", ha='center', va='center',
                        fontsize=10, color='grey',
                    )
                    continue
                delta_pct = (
                    (val - bl) / abs(bl) * 100 if bl != 0 else 0.0
                )
                sign = "+" if delta_pct >= 0 else ""
                ax.text(
                    i, j,
                    f"{val:.4f}\n{sign}{delta_pct:.1f}%",
                    ha='center', va='center', fontsize=8,
                    fontweight='bold',
                )

        ax.set_xticks(range(B_LIQ))
        ax.set_xticklabels(tick_labels)
        ax.set_yticks(range(B_VOL))
        ax.set_yticklabels(tick_labels)
        ax.set_xlabel('Liquidity')
        ax.set_ylabel('Volatility')
        ax.set_title(f"{ptitle}  (baseline = {bl:.5f})", fontsize=11)
        cbar = fig.colorbar(im, ax=ax, shrink=0.8)
        cbar.set_label('\u0394 Fitness vs Baseline')

    fig.suptitle(title, fontsize=14, fontweight='bold')
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved: {save_path}")
    plt.show()


def print_results_table(results_df, B_LIQ=3, B_VOL=3, label="Specialist"):
    """
    Print a per-cell results table.
    *label* is shown in the header (e.g. "Specialist" or "Ensemble").
    """
    vol_labels = ["Low", "Medium", "High"][:B_VOL]
    liq_labels = ["Low", "Medium", "High"][:B_LIQ]

    header = (f"\n{'Vol':<8} {'Liq':<8} {'Baseline':>10} "
              f"{label:>10} {'Δ%':>8} {'Best':>6} {'N':>7}")
    print(header)
    print("=" * len(header.strip()))

    w_spec, w_base, w_ens, total_n = 0.0, 0.0, 0.0, 0
    for j in range(B_VOL):
        for i in range(B_LIQ):
            r = results_df[
                (results_df.liq_bin == i) & (results_df.vol_bin == j)
            ]
            if r.empty:
                continue
            r = r.iloc[0]
            spec = r.specialist_fitness
            base = r.baseline_fitness
            n = int(r.n_orders)

            if np.isnan(spec) and np.isnan(base):
                print(f"{vol_labels[j]:<8} {liq_labels[i]:<8} "
                      f"{'—':>10} {'—':>10} {'—':>8} {'—':>6} {n:>7}")
                continue

            ens = max(
                spec if not np.isnan(spec) else -np.inf,
                base if not np.isnan(base) else -np.inf,
            )
            delta_pct = (
                (spec - base) / abs(base) * 100
                if (not np.isnan(spec) and not np.isnan(base)
                    and base != 0)
                else np.nan
            )
            d_str = (
                f"{delta_pct:+.1f}%" if not np.isnan(delta_pct) else "—"
            )
            best = "spec" if (not np.isnan(spec) and spec >= ens) else "base"

            b_str = f"{base:.5f}" if not np.isnan(base) else "—"
            s_str = f"{spec:.5f}" if not np.isnan(spec) else "—"

            print(f"{vol_labels[j]:<8} {liq_labels[i]:<8} "
                  f"{b_str:>10} {s_str:>10} {d_str:>8} {best:>6} {n:>7}")

            if not np.isnan(spec):
                w_spec += spec * n
            if not np.isnan(base):
                w_base += base * n
            w_ens += ens * n
            total_n += n

    print("-" * len(header.strip()))
    if total_n:
        ov_s = w_spec / total_n
        ov_b = w_base / total_n
        ov_e = w_ens / total_n
        ov_i = (ov_s - ov_b) / abs(ov_b) * 100 if ov_b else 0
        ov_ei = (ov_e - ov_b) / abs(ov_b) * 100 if ov_b else 0
        print(f"{'Wt-Avg':<8} {'':8} {ov_b:>10.5f} {ov_s:>10.5f} "
              f"{ov_i:>+7.1f}% {'':>6} {total_n:>7}")
        print(f"{'Ensembl':<8} {'':8} {'':>10} {ov_e:>10.5f} "
              f"{ov_ei:>+7.1f}% {'':>6} {total_n:>7}")


def evaluate_archive_on_orders(archive, env, baseline_model, orders_df,
                                cell2indices, B_LIQ=3, B_VOL=3, vecnorm=None):
    """
    For each cell: run specialist + baseline on phenotype-matched orders.
    Uses VecNormPolicyWrapper to swap snapshots in/out of a single model.
    """
    wrapper = VecNormPolicyWrapper(baseline_model, vecnorm)
    wrapper.policy.eval()
    baseline_snap = snapshot_policy(baseline_model)

    results = []

    for i in range(B_LIQ):
        for j in range(B_VOL):
            idxs = cell2indices.get((i, j), [])
            cell = archive.grid[i][j]

            if len(idxs) == 0 or cell is None:
                results.append(dict(
                    liq_bin=i, vol_bin=j, n_orders=len(idxs),
                    specialist_fitness=np.nan, baseline_fitness=np.nan,
                    improvement_pct=np.nan))
                continue

            # Specialist
            wrapper.load_policy_snapshot(cell["snapshot"], device="cpu")
            spec_orders = env.execute_orders_vectorized(
                wrapper, idxs, collect_step_info=True,
                show_progress=False, model_name=f"ME@{i}-{j}")
            spec_fit = _fitness_from_orders(spec_orders)

            # Baseline on same orders
            wrapper.load_policy_snapshot(baseline_snap, device="cpu")
            base_orders = env.execute_orders_vectorized(
                wrapper, idxs, collect_step_info=True,
                show_progress=False, model_name=f"base@{i}-{j}")
            base_fit = _fitness_from_orders(base_orders)

            imp = ((spec_fit - base_fit) / abs(base_fit) * 100
                   if base_fit != 0 and not np.isnan(base_fit) else np.nan)

            results.append(dict(
                liq_bin=i, vol_bin=j, n_orders=len(idxs),
                specialist_fitness=spec_fit, baseline_fitness=base_fit,
                improvement_pct=imp))

            print(f"  ({i},{j}): spec={spec_fit:.5f} base={base_fit:.5f} "
                  f"Δ={imp:+.1f}% n={len(idxs)}")

    # Restore baseline
    wrapper.load_policy_snapshot(baseline_snap, device="cpu")
    return pd.DataFrame(results)


def build_ensemble_grid(results_df, B_LIQ=3, B_VOL=3):
    """
    Build a fitness grid using the best of specialist or baseline per cell.
    """
    grid = np.full((B_LIQ, B_VOL), np.nan)
    for _, row in results_df.iterrows():
        i, j = int(row.liq_bin), int(row.vol_bin)
        spec = row.specialist_fitness
        base = row.baseline_fitness
        if np.isnan(spec) and np.isnan(base):
            continue
        elif np.isnan(spec):
            grid[i, j] = base
        elif np.isnan(base):
            grid[i, j] = spec
        else:
            grid[i, j] = max(spec, base)
    return grid


def execute_ensemble_orders(
    archive, env, baseline_model, orders_df, order_indices,
    test_results, B_LIQ=3, B_VOL=3, vecnorm=None,
):
    """
    Execute orders using a best-of-specialist/baseline ensemble.
    Each order is routed to whichever model (specialist or baseline)
    had higher fitness in its (liq_bin, vol_bin) cell during evaluation.

    Returns a list of order traces in the same positional order as
    *order_indices*, compatible with the TCA pipeline.
    """
    use_specialist = {}
    for _, row in test_results.iterrows():
        i, j = int(row.liq_bin), int(row.vol_bin)
        spec = row.specialist_fitness
        base = row.baseline_fitness
        if np.isnan(spec) or archive.grid[i][j] is None:
            use_specialist[(i, j)] = False
        elif np.isnan(base):
            use_specialist[(i, j)] = True
        else:
            use_specialist[(i, j)] = spec > base

    wrapper = VecNormPolicyWrapper(baseline_model, vecnorm)
    wrapper.policy.eval()
    baseline_snap = snapshot_policy(baseline_model)

    orders_view = (
        orders_df[["liq_norm", "vol_norm"]]
        .fillna(0.0)
        .to_numpy(dtype=float)
    )

    cell_groups = {}
    for pos, idx in enumerate(order_indices):
        liq, vol = orders_view[idx]
        i = _to_bin(liq, B_LIQ)
        j = _to_bin(vol, B_VOL)
        cell_groups.setdefault((i, j), []).append((pos, idx))

    results = [None] * len(order_indices)

    n_spec_cells = sum(1 for v in use_specialist.values() if v)
    n_base_cells = sum(1 for v in use_specialist.values() if not v)
    print(f"Ensemble routing: {n_spec_cells} specialist cells, "
          f"{n_base_cells} baseline cells")

    for (i, j), pos_idx_pairs in tqdm(
        cell_groups.items(), desc="Ensemble execution"
    ):
        positions = [p for p, _ in pos_idx_pairs]
        idxs = [idx for _, idx in pos_idx_pairs]

        if use_specialist.get((i, j), False):
            snap = archive.grid[i][j]["snapshot"]
            wrapper.load_policy_snapshot(snap, device="cpu")
            tag = f"ens_spec@{i}-{j}"
        else:
            wrapper.load_policy_snapshot(baseline_snap, device="cpu")
            tag = f"ens_base@{i}-{j}"

        cell_orders = env.execute_orders_vectorized(
            wrapper, idxs, collect_step_info=True,
            show_progress=False, model_name=tag,
        )
        for pos, trace in zip(positions, cell_orders):
            results[pos] = trace

    wrapper.load_policy_snapshot(baseline_snap, device="cpu")
    return results


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

    # --- Illumination loop (cross-cell parent sampling) ---
    logging.info(f"Illumination loop: iterating {iters} times")
    for _ in tqdm(range(iters), desc="Illumination loop: iterating over iterations",position=1, leave=False):
        occupied = [
            (ii, jj) for ii in range(B_LIQ) for jj in range(B_VOL)
            if archive.grid[ii][jj] is not None
        ]
        targets = random.choices(nonempty_cells, k=min(children_per_iter, len(nonempty_cells)))
        for (i, j) in targets:
            pi, pj = random.choice(occupied)
            parent_snap = archive.grid[pi][pj]["snapshot"]
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
    
    # Illumination loop with progress tracking (cross-cell parent sampling)
    print(f"  Illumination: {iters} iterations × {children_per_iter} children")
    for iter_idx in tqdm(range(iters), desc="  Illumination", leave=False):
        occupied = [
            (ii, jj) for ii in range(B_LIQ) for jj in range(B_VOL)
            if archive.grid[ii][jj] is not None
        ]
        targets = random.choices(nonempty_cells, k=min(children_per_iter, len(nonempty_cells)))
        
        improvements = 0
        for (i, j) in targets:
            pi, pj = random.choice(occupied)
            parent_snap = archive.grid[pi][pj]["snapshot"]
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
    
    # Illumination with progress bar (cross-cell parent sampling)
    for iter_idx in tqdm(range(iters), desc="  MAP-Elites iterations"):
        occupied = [
            (ii, jj) for ii in range(B_LIQ) for jj in range(B_VOL)
            if archive.grid[ii][jj] is not None
        ]
        targets = random.choices(nonempty_cells, k=min(children_per_iter, len(nonempty_cells)))
        for (i, j) in targets:
            pi, pj = random.choice(occupied)
            parent_snap = archive.grid[pi][pj]["snapshot"]
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