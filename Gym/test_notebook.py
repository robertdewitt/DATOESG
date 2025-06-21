#!/usr/bin/env python
"""test_notebook.py
A lightweight script version of the original Jupyter notebook used to train/evaluate
`MultiOrderExecutionEnv` and visualise order execution results.

Key features
------------
1. Collects market data until the full time‐horizon, even after the order has completed, by
   continuing to step the environment with a neutral action (0).
2. Safely accesses info‐dict keys that disappear after completion (e.g. `shares_remaining`).
3. Plots:
   • Prices & reward (Fill, VWAP, Mid, Order VWAP, Total Reward)
   • Shares remaining, trade size and market volume
   • Action %, accumulated impact
4. Marks the completion time with a vertical dotted line.

Run this script directly or copy/paste cells into a Jupyter notebook.
"""

# ---------------------------------------------------------------------------
# Imports & setup
# ---------------------------------------------------------------------------
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from mkt_data_yfinance import MarketDataLoader as mdl
from optimal_execution_env import MultiOrderExecutionEnv
from order_generator import OrderGenerator
from notebook_logging_setup import quick_setup
from gymnasium.wrappers import TimeLimit
from gymnasium.envs.registration import register
import gymnasium as gym
from stable_baselines3 import PPO

# Less verbose logging
quick_setup(debug=False)

# ---------------------------------------------------------------------------
# Data & environment initialisation
# ---------------------------------------------------------------------------
max_trading_rate = 0.8
max_order_size_in_adv_pct = 0.1
num_training_orders = 50_000  # adjust as required
num_eval_episodes = 10        # number of orders to evaluate/plot

# Load market data once
TICKERS = ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'TSLA', 'NVDA', 'JPM', 'V', 'JNJ', 'DIS']
md_loader = mdl()
stock_df_list = md_loader.load_data(TICKERS)
stock_df_list = md_loader.pre_process_data(stock_df_list)

# Generate synthetic orders for training/evaluation
orders_df = OrderGenerator(
    stock_df_list=stock_df_list,
    market_data=md_loader,
    debug=False,
    max_adv_pct=max_order_size_in_adv_pct,
    num_orders=num_training_orders,
).get_orders()

# ---------------------------------------------------------------------------
# Environment registration & creation
# ---------------------------------------------------------------------------
register(
    id="MultiOrderExecutionEnv-v0",
    entry_point="optimal_execution_env:MultiOrderExecutionEnv",
    max_episode_steps=None,
)

# Determine global max horizon to wrap with TimeLimit
max_horizon = int(orders_df["time_horizon"].max())

base_env = gym.make(
    "MultiOrderExecutionEnv-v0",
    stock_df_list=stock_df_list,
    orders_df=orders_df,
    impact_coef=0.01,
    decay_rate=0.1,
    num_envs=1,
    window_size=1,
    min_rate=0,
    max_rate=max_trading_rate,
)

# Time-limit wrapper ensures episodes always terminate at max_horizon
env = TimeLimit(base_env, max_episode_steps=max_horizon)

# ---------------------------------------------------------------------------
# Agent (load a pre-trained PPO model or train a small one for demo)
# ---------------------------------------------------------------------------
MODEL_PATH = "ppo_multiorder_exec.zip"
try:
    model = PPO.load(MODEL_PATH, env=base_env)
    print("Loaded existing PPO model.")
except FileNotFoundError:
    print("Training a quick PPO model – this may take a while on CPU…")
    model = PPO("MlpPolicy", base_env, verbose=0, learning_rate=3e-4)
    model.learn(total_timesteps=200_000)
    model.save(MODEL_PATH)
    print("Model trained & saved.")

# ---------------------------------------------------------------------------
# Evaluation loop – collect full-horizon data
# ---------------------------------------------------------------------------
orders = []  # list of per-episode info-dict sequences

for ep in range(num_eval_episodes):
    obs, info = base_env.reset()
    done, truncated = False, False
    order_info = []  # list of per-step info dicts
    total_reward = 0.0

    while not truncated:
        # After order completion (done), keep stepping with neutral action 0
        if done:
            action = 0
        else:
            action, _ = model.predict(obs, deterministic=True)

        obs, reward, done, truncated, info = base_env.step(action)
        order_info.append(info)
        total_reward += reward

    print(f"Episode {ep+1} finished – total reward {total_reward:.2f}")
    orders.append(order_info)

base_env.close()

# ---------------------------------------------------------------------------
# Plotting helper
# ---------------------------------------------------------------------------

def plot_order(order_info: list[dict], idx: int = 0):
    """Plot price & execution metrics for a single order_info sequence."""
    if not order_info:
        print("Empty order_info – nothing to plot.")
        return

    md0 = order_info[0]  # first step meta
    ticker = md0.get("ticker", "NA")
    side = md0.get("side", "NA").capitalize()
    horizon = md0.get("time_horizon", len(order_info) - 1)
    total_shares = md0.get("order_qty", md0.get("shares_remaining", 0) + md0.get("last_trade_size", 0))

    # Extract series (using .get so keys can be missing after completion)
    times = [x["current_step"] for x in order_info]
    fill_prices = [x.get("last_fill_price", np.nan) for x in order_info]
    vwap_prices = [x.get("vwap_price", np.nan) for x in order_info]
    mid_prices = [x.get("mid_price", np.nan) for x in order_info]
    order_vwap = [x.get("order_vwap", np.nan) for x in order_info]
    rewards = [x.get("total_reward", np.nan) for x in order_info]

    shares_rem = [x.get("shares_remaining", np.nan) for x in order_info]
    trade_sizes = [x.get("last_trade_size", 0) for x in order_info]
    volumes = [x.get("current_volume", np.nan) for x in order_info]

    action_pct = [x.get("action_percentage", 0.0) * 100 for x in order_info]
    acc_impact = [x.get("accumulated_impact", np.nan) for x in order_info]

    # Figure & subplots
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(12, 12), sharex=True)
    fig.suptitle(
        f"[{ticker} | {side}] Order {idx+1}: {total_shares:.0f} shares (Horizon {max(times)} min)",
        fontsize=14,
    )

    # --- Prices panel ---
    ax1.set_ylabel("Price")
    ax1.plot(times, fill_prices, label="Fill", color="blue")
    ax1.plot(times, vwap_prices, label="VWAP", color="green")
    ax1.plot(times, mid_prices, label="Mid", color="gray")
    ax1.plot(times, order_vwap, label="Order VWAP", color="purple", linestyle="--")
    ax1.grid(True)

    ax1r = ax1.twinx()
    ax1r.set_ylabel("Reward", color="red")
    ax1r.plot(times, rewards, label="Total Reward", color="red", linestyle=":")

    lines, labels = ax1.get_legend_handles_labels()
    lines2, labels2 = ax1r.get_legend_handles_labels()
    ax1.legend(lines + lines2, labels + labels2, loc="best")

    # --- Shares/trade size & volume panel ---
    ax2.set_ylabel("Shares Remaining", color="orange")
    ax2.plot(times, shares_rem, label="Shares Remaining", color="orange", linestyle="--")
    ax2.grid(True)

    ax2r = ax2.twinx()
    ax2r.set_ylabel("Trade Size / Volume", color="red")
    ax2r.scatter(times, trade_sizes, label="Trade Size", color="red", marker="x")
    ax2r.plot(times, volumes, label="Volume", color="teal", linestyle="-.")

    lines, labels = ax2.get_legend_handles_labels()
    lines2, labels2 = ax2r.get_legend_handles_labels()
    ax2.legend(lines + lines2, labels + labels2, loc="best")

    # --- Action % & impact panel ---
    ax3.set_ylabel("Action %", color="green")
    ax3.set_xlabel("Time (minutes)")
    ax3.plot(times, action_pct, label="Action %", color="green")
    ax3.grid(True)

    ax3r = ax3.twinx()
    ax3r.set_ylabel("Accumulated Impact", color="purple")
    ax3r.plot(times, acc_impact, label="Accum Impact", color="purple", linestyle="--")

    lines, labels = ax3.get_legend_handles_labels()
    lines2, labels2 = ax3r.get_legend_handles_labels()
    ax3.legend(lines + lines2, labels + labels2, loc="best")

    # X-axis full horizon
    ax3.set_xlim(0, max(times))

    # Vertical line for completion
    if any(x.get("shares_remaining", None) == 0 for x in order_info):
        compl_time = next(x["current_step"] for x in order_info if x.get("shares_remaining", None) == 0)
        for ax in (ax1, ax2, ax3):
            ax.axvline(x=compl_time, color="gray", linestyle=":", alpha=0.5)

    plt.tight_layout()
    plt.show()


# ---------------------------------------------------------------------------
# Plot first few orders
# ---------------------------------------------------------------------------
for idx, info_seq in enumerate(orders[:num_eval_episodes]):
    plot_order(info_seq, idx)


# In[14]:


# TODO: discretise actions
# TODO: get 4090 gpu working
# TODO: clean plot sample order results  - add action percent versus accumulated impact, add reward
# TODO: is it minimizing reward correctly?
# TODO: add accumulated impact as a feature to allow it to learn to avoid it where possible (ot total cost)
# TODO: make sure decisions are based on the prior bin's information for price and current bin for volume
# TODO: orders are not completing execution even though the order size is small enough
# TODO: add hypothetical slippage to the ultimate best price for the horizon to indicate opportunity cost
# TODO: calculate the theoretical slippage of all orders 
# TODO: paralellize and vectorize
# TODO: get GPU working
# TODO: fix the warnings in check env
# TODO: run across multiple agents
# TODO: sort out across dates# 
# TODO: how to improve the state space such that the agent can learn to execute orders better?
# TODO: add more orders, more 
# TODO: could we use unfilled quanity/EHV as a penelty to ensure it doesn't get too far behind so it can complete?
# TODO: make volume profiles to scale volume and remove auctions?
# TODO: include regimes?


# In[ ]:




