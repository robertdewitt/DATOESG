"""
Map Elites implementation for the optimal execution environment

First cut
"""
import numpy as np, torch, torch.nn as nn, torch.nn.functional as F
from copy import deepcopy
import logging
from tqdm import tqdm
import os, json, math, pickle, numpy as np, matplotlib.pyplot as plt
from dataclasses import dataclass
from typing import List, Tuple, Optional

OBS_DIM, N_ACTIONS = 15, 11

class LinearPolicy(nn.Module):
    def __init__(self, obs_dim=OBS_DIM, hidden=64, device="cpu"):
        super().__init__()
        self.device = torch.device(device)
        self.net = nn.Sequential(
            nn.LayerNorm(obs_dim),
            nn.Linear(obs_dim, hidden),
            nn.Tanh(),
            nn.Linear(hidden, N_ACTIONS),
        )
        self.eval()

    @torch.no_grad()
    def predict(self, obs: np.ndarray, deterministic: bool = True):
        x = torch.tensor(obs, dtype=torch.float32, device=self.device).unsqueeze(0)
        logits = self.net(x)
        if deterministic:
            a = torch.argmax(logits, dim=-1).item()
        else:
            probs = F.softmax(logits, dim=-1)
            a = torch.multinomial(probs, 1).item()
        return int(a), None

    def mutate(self, sigma=0.05):
        with torch.no_grad():
            for p in self.parameters():
                p.add_(sigma * torch.randn_like(p))


def _desc_from_trace(order_trace):
    """
    This function is used to calculate the finish_ratio and smoothness of a policy.

    completion time / horizon; and smoothness = 1/(1+std(|action_percentage|))

    Args:
        order_trace: List of order execution data (list of lists of step dictionaries)

    Returns:
        Tuple of finish_ratio and smoothness
    """
    
    if not order_trace:
        return 1.0, 0.0
    horizon = max(x["time_horizon"] for x in order_trace)
    finished_at = None
    parts = []
    for t, x in enumerate(order_trace):
        parts.append(abs(x.get("action_percentage", 0.0)))
        if x.get("shares_remaining", 1) <= 0 and finished_at is None:
            finished_at = t + 1
    if finished_at is None:
        finished_at = horizon
    finish_ratio = float(np.clip(finished_at / max(1, horizon), 0.0, 1.0))
    smoothness  = float(1.0 / (1.0 + (np.std(parts) if len(parts) else 0.0)))
    return finish_ratio, smoothness


def _fitness_from_orders(orders):
    # maximize negative mean total cost
    costs = []
    for tr in orders:
        if not tr: 
            continue
        # fall back if per-step total not present in collected trace; use 'total_cost' snapshot
        step_costs = [s.get("total_step_cost") for s in tr if "total_step_cost" in s]
        if step_costs:
            costs.append(np.sum(step_costs))
        else:
            # vectorized path exposes 'total_cost' snapshot per collected step; take final snapshot
            costs.append(tr[-1].get("total_cost", 0.0))
    return -np.mean(costs) if costs else -np.inf



class Archive:
    def __init__(self, bins_f=20, bins_s=20):
        self.bf, self.bs = bins_f, bins_s
        self.grid = [[None for _ in range(self.bs)] for _ in range(self.bf)]
    def _coords(self, f, s):
        return min(self.bf-1, int(f*self.bf)), min(self.bs-1, int(s*self.bs))
    def insert(self, policy, fitness, desc):
        i, j = self._coords(*desc)
        cell = self.grid[i][j]
        if (cell is None) or (fitness > cell["fitness"]):
            self.grid[i][j] = {"policy": deepcopy(policy).cpu(), "fitness": fitness, "desc": desc}
    def sample(self, k=10):
        cells = [c for row in self.grid for c in row if c is not None]
        if not cells: return []
        idx = np.random.choice(len(cells), size=min(k, len(cells)), replace=False)
        return [cells[m]["policy"] for m in idx]
    def best(self):
        best = None
        for row in self.grid:
            for c in row:
                if c and (best is None or c["fitness"] > best["fitness"]):
                    best = c
        return best



def map_elites(env, init_policies=64, iters=50, eval_orders_per_policy=4,
               mutate_sigma=0.05, device="cpu", show_progress=False,
               fixed_order_indices: Optional[List[int]] = None,
               bins_f=20, bins_s=20):
    logging.getLogger("root").setLevel(logging.WARNING)
    archive = Archive(bins_f, bins_s)

    def _pick_indices():
        if fixed_order_indices is None:
            return env.generate_random_order_indices(eval_orders_per_policy)
        # strictly fair: same slice every time
        if len(fixed_order_indices) < eval_orders_per_policy:
            raise ValueError("fixed_order_indices shorter than eval_orders_per_policy")
        return list(fixed_order_indices[:eval_orders_per_policy])

    def evaluate(pi, tag="cand"):
        idxs = _pick_indices()
        orders = env.execute_orders_vectorized(
            pi, idxs, collect_step_info=True, show_progress=False, model_name=tag
        )
        fit = _fitness_from_orders(orders)
        desc = _desc_from_trace(orders[0] if orders else [])
        return fit, desc

    for i in tqdm(range(init_policies), desc="MAP-Elites: seeding"):
        pi = LinearPolicy(device=device)
        f, d = evaluate(pi, tag=f"seed-{i:04d}")
        if np.isfinite(f):
            archive.insert(pi, f, d)

    for it in tqdm(range(iters), desc="MAP-Elites: iterations"):
        parents = archive.sample(k=max(1, init_policies // 2)) or [LinearPolicy(device=device)]
        for j, p in enumerate(parents):
            c = deepcopy(p).to(device)
            c.mutate(mutate_sigma)
            f, d = evaluate(c, tag=f"iter{it:03d}-child{j:03d}")
            if np.isfinite(f):
                archive.insert(c, f, d)

    return archive


@dataclass
class EliteEntry:
    i: int
    j: int
    fitness: float
    finish_ratio: float
    smoothness: float
    path: Optional[str] = None  # filled when saved to disk




def archive_to_matrix(archive) -> Tuple[np.ndarray, np.ndarray]:
    """Return (fitness_grid, mask) with NaN where empty."""
    H, W = len(archive.grid), len(archive.grid[0])
    M = np.full((H, W), np.nan, dtype=float)
    for i in range(H):
        for j in range(W):
            cell = archive.grid[i][j]
            if cell is not None and np.isfinite(cell["fitness"]):
                M[i, j] = float(cell["fitness"])
    return M, ~np.isnan(M)

def plot_archive_heatmap(archive, title="Archive fitness (higher is better)"):
    M, mask = archive_to_matrix(archive)
    plt.figure(figsize=(7, 5))
    # default colormap; no explicit colors so it's portable
    im = plt.imshow(M, origin="lower", aspect="auto")
    plt.colorbar(im, label="fitness")
    plt.xlabel("smoothness bins (low → high)")
    plt.ylabel("finish_ratio bins (early → late)")
    plt.title(title)
    plt.tight_layout()
    plt.show()

def select_elites(archive,
                  finish_range: Tuple[float,float]=(0.0,1.0),
                  smooth_range: Tuple[float,float]=(0.0,1.0),
                  top_k: int = 5) -> List[EliteEntry]:
    out: List[EliteEntry] = []
    H, W = len(archive.grid), len(archive.grid[0])
    for i in range(H):
        for j in range(W):
            cell = archive.grid[i][j]
            if cell is None: 
                continue
            fr, sm = cell["desc"]
            if (finish_range[0] <= fr <= finish_range[1]) and (smooth_range[0] <= sm <= smooth_range[1]):
                out.append(EliteEntry(i=i, j=j, fitness=float(cell["fitness"]),
                                      finish_ratio=float(fr), smoothness=float(sm)))
    out.sort(key=lambda e: e.fitness, reverse=True)
    return out[:top_k]