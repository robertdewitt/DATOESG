"""
Map Elites implementation for the optimal execution environment

First cut
"""
import numpy as np, torch, torch.nn as nn, torch.nn.functional as F
from copy import deepcopy
import logging

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
    # completion time / horizon; and smoothness = 1/(1+std(|action_percentage|))
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


def map_elites(env, init_policies=64, iters=50, eval_orders_per_policy=4, mutate_sigma=0.05, device="cpu", show_progress=False, fixed_order_indices=None):
    arch = Archive(20, 20)

    logging.getLogger("optimal_execution_env_vectorized").setLevel(logging.WARNING)


    def evaluate(pi):
        # vectorized evaluation over random orders
        if fixed_order_indices is None:
            idxs = env.generate_random_order_indices(eval_orders_per_policy)
        else:
            # pick random orders from fixed 
            idxs = np.random.choice(fixed_order_indices, size=eval_orders_per_policy, replace=False)
        orders = env.execute_orders_vectorized(pi, idxs, collect_step_info=True, show_progress=False)
        fit = _fitness_from_orders(orders)
        desc = _desc_from_trace(orders[0] if len(orders) > 0 else [])
        return fit, desc

    # seeding
    logging.info(f"Seeding {init_policies} policies")
    for _ in range(init_policies):
        pi = LinearPolicy(device=device)
        f, d = evaluate(pi)
        if np.isfinite(f): arch.insert(pi, f, d)


    logging.info(f"Illuminating {iters} iterations, mutate_sigma={mutate_sigma}")
    # illumination
    for it in range(iters):
        parents = arch.sample(k=max(1, init_policies//2)) or [LinearPolicy(device=device)]
        for p in parents:
            c = deepcopy(p).to(device)
            c.mutate(mutate_sigma)
            f, d = evaluate(c)
            if np.isfinite(f): arch.insert(c, f, d)
        best = arch.best()
        if best:
            print(f"[{it+1}] best_fitness={best['fitness']:.6f}, desc={best['desc']}")

    return arch
