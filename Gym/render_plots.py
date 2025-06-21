import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import logging

# Set up logger for this module
logger = logging.getLogger(__name__)


def plot_orders(orders, num_orders=3):
    """
    Plots the order information including prices, quantities, and action percentages.
    @param orders: List of order information dictionaries.
    Each dictionary should contain keys like 'ticker', 'side', 'time_horizon', 'order_qty',
    'current_step', 'mid_price', 'immediate_impact', 'action_percentage', 'accumulated_impact',
    'last_fill_price', 'vwap_price', 'order_vwap', 'total_reward', 'last_trade_size', and 'shares_remaining'.
    @return: None
    """
    for i, order_info in enumerate(orders[:num_orders]):
        # Extract meta data for this order
        ticker = order_info[0]["ticker"]
        side = order_info[0]["side"]
        time_horizon = order_info[0]["time_horizon"]
        adv_val = order_info[0].get("adv", None)

        if 'order_qty' in order_info[0]:
            total_shares = order_info[0]['order_qty']
        else:
            total_shares = (order_info[0].get('shares_remaining', 0) +
                          order_info[0].get('last_trade_size', 0))

        # Create figure with 3 subplots
        fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(12, 12), sharex=True)
        
        # Set title for the entire figure
        subtitle_parts = [
            f"[{ticker} | {side.capitalize()}] Order {i+1}",
            f"Total {total_shares:,.0f} Shares",
            f"Horizon: {time_horizon}"
        ]
        if adv_val is not None:
            subtitle_parts.append(f"ADV: {adv_val:,.0f}")
        fig.suptitle(" | ".join(subtitle_parts), fontsize=14)

        # Plot 1: Prices
        ax1.set_ylabel("Price", color="blue")
        times = [x["current_step"] for x in order_info]
        mid_prices = [x["mid_price"] for x in order_info]
        
        # Convert to bps and percentages
        immediate_impact = [x.get("immediate_impact", 0) * 10000 for x in order_info]
        action_percentages = [x.get("action_percentage", 0) * 100 for x in order_info]
        accumulated_impacts = [x.get("accumulated_impact", np.nan) * 10000 for x in order_info]

        fill_prices = [x.get("last_fill_price", np.nan) for x in order_info]
        vwap_prices = [x.get("vwap_price", np.nan) for x in order_info]
        order_vwap = [x.get("order_vwap", np.nan) for x in order_info]
        total_rewards = [x.get("total_reward", np.nan) for x in order_info]

        ax1.plot(times, fill_prices, label="Fill Price", color="blue")
        ax1.plot(times, vwap_prices, label="Market VWAP Price", color="green")
        ax1.plot(times, order_vwap, label="Order VWAP Price", color="purple", linestyle="--")
        ax1.grid(True)
       
        # Add reward on secondary y-axis
        ax1_reward = ax1.twinx()
        ax1_reward.set_ylabel("Total Reward", color="red")
        ax1_reward.plot(times, total_rewards, label="Total Reward", color="red", linestyle=":")
        
        # Combine legends
        lines1, labels1 = ax1.get_legend_handles_labels()
        lines2, labels2 = ax1_reward.get_legend_handles_labels()
        ax1.legend(lines1 + lines2, labels1 + labels2, loc="best")

        # Plot 2: Quantities
        ax2.set_ylabel("Quantity", color="red")
        trade_sizes = [x["last_trade_size"] for x in order_info]
        shares_remaining = [x["shares_remaining"] for x in order_info]

        ax2.plot(times, shares_remaining, label="Shares Remaining", color="orange", linestyle="--")
        ax2.grid(True)
          
        # Add trade size on secondary y-axis
        ax2_trade = ax2.twinx()
        ax2_trade.set_ylabel("Trade Size", color="red")
        ax2_trade.scatter(times, trade_sizes, label="Trade Size", color="red", marker="o")
        
        # Combine legends
        lines1, labels1 = ax2.get_legend_handles_labels()
        lines2, labels2 = ax2_trade.get_legend_handles_labels()
        ax2.legend(lines1 + lines2, labels1 + labels2, loc="best")

        # Plot 3: Action Percentage
        ax3.set_ylabel("Action %", color="green")
        ax3.set_xlabel("Time (minutes)")
        action_percentages = [x.get("action_percentage", 0) * 100 for x in order_info]
        ax3.plot(times, action_percentages, label="Action %", color="green")
        ax3.grid(True)
        
        # Add accumulated impact on secondary y-axis
        ax3_impact = ax3.twinx()
        ax3_impact.set_ylabel("Accumulated Impact (bps)", color="purple")
        ax3_impact.plot(times, accumulated_impacts, label="Accumulated Impact (bps)",
                       color="purple", linestyle="--")
        ax3_impact.scatter(times, immediate_impact, label="Immediate Impact (bps)",
                          color="brown", marker="x")
        
        # Combine legends
        lines1, labels1 = ax3.get_legend_handles_labels()
        lines2, labels2 = ax3_impact.get_legend_handles_labels()
        ax3.legend(lines1 + lines2, labels1 + labels2, loc="best")

        # Set x-axis limits to show full horizon
        ax1.set_xlim(0, time_horizon)
        
        # Add vertical line at order completion time
        if any((x.get("shares_remaining", None) == 0) for x in order_info):
            completion_time = next(x["current_step"] for x in order_info if x.get("shares_remaining", None) == 0)
            for ax in [ax1, ax2, ax3]:
                ax.axvline(x=completion_time, color='gray', linestyle=':', alpha=0.5,
                          label='Order Completed')
                ax.legend()

        # Final layout/visualise
        plt.tight_layout()
        plt.show()


def display_order_info(orders_df, num_orders=3, name="Training"):
    """
    Displays order information in a readable format.
    @param orders: List of order information dictionaries.
    Each dictionary should contain keys like 'ticker', 'side', 'time_horizon', 'order_qty',
    'current_step', 'mid_price', 'immediate_impact', 'action_percentage', 'accumulated_impact',
    'last_fill_price', 'vwap_price', 'order_vwap', 'total_reward', 'last_trade_size', and 'shares_remaining'.
    @return: None
    """
    print(f"{num_orders} {name} Orders:")
    print("========================================")
    i = 1
    num_orders = min(num_orders, len(orders_df))
    
    # Display each order in the DataFrame
    if num_orders == 0:
        print("No orders to display.")
        return
    
    for index, row in orders_df.iterrows():
        if i > num_orders:
            break
            
        # Print order details
        print(f"{name} ({i}):", end=' ')
        for key in row.keys():
            val = row[key]
            if key == 'adv':
                # Handle potential Series/array stored in the cell
                if isinstance(val, pd.Series):
                    val_out = val.iloc[0]
                elif isinstance(val, (list, np.ndarray)):
                    val_out = val[0]
                else:
                    val_out = val
                # Some objects might still be nan; guard before formatting
                try:
                    print(f"{key}: {float(val_out):,.0f}", end=' ')
                except (ValueError, TypeError):
                    print(f"{key}: {val_out}", end=' ')
            elif key == 'adv_pct':
                print(f"{key}: {val:.2%}", end=' ')
            else:
                print(f"{key}: {val}", end=' ')
        print()
        i += 1
        if i > num_orders:
            break
    print("========================================")


def plot_order_histograms(train_df, test_df=None, date_col='Datetime'):
    """
    Plots histograms of order characteristics comparing train and test distributions.
    @param train_df: DataFrame containing training order information.
    @param test_df: Optional DataFrame containing test order information.
    @param date_col: Column name containing the date/time information.
    @return: None
    """
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    axes = axes.flatten()

    # Plot histograms for each characteristic
    plot_idx = 0
    
    # Order size distribution
    axes[plot_idx].hist(train_df['order_qty'], bins=50, alpha=0.5, label='Train')
    if test_df is not None:
        axes[plot_idx].hist(test_df['order_qty'], bins=50, alpha=0.5, label='Test')
    axes[plot_idx].set_title('Order Size Distribution')
    axes[plot_idx].set_xlabel('Order Size')
    axes[plot_idx].set_ylabel('Frequency')
    axes[plot_idx].legend()
    plot_idx += 1

    # Time horizon distribution
    axes[plot_idx].hist(train_df['time_horizon'], bins=50, alpha=0.5, label='Train')
    if test_df is not None:
        axes[plot_idx].hist(test_df['time_horizon'], bins=50, alpha=0.5, label='Test')
    axes[plot_idx].set_title('Time Horizon Distribution')
    axes[plot_idx].set_xlabel('Time Horizon (minutes)')
    axes[plot_idx].set_ylabel('Frequency')
    axes[plot_idx].legend()
    plot_idx += 1

    # ADV percentage distribution
    axes[plot_idx].hist(train_df['adv_pct'], bins=50, alpha=0.5, label='Train')
    if test_df is not None:
        axes[plot_idx].hist(test_df['adv_pct'], bins=50, alpha=0.5, label='Test')
    axes[plot_idx].set_title('ADV Percentage Distribution')
    axes[plot_idx].set_xlabel('ADV Percentage')
    axes[plot_idx].set_ylabel('Frequency')
    axes[plot_idx].legend()
    plot_idx += 1

    # Side distribution (buy/sell)
    train_sides = train_df['side'].value_counts()
    axes[plot_idx].bar(train_sides.index, train_sides.values, alpha=0.5, label='Train')
    if test_df is not None:
        test_sides = test_df['side'].value_counts()
        axes[plot_idx].bar(test_sides.index, test_sides.values, alpha=0.5, label='Test')
    axes[plot_idx].set_title('Order Side Distribution')
    axes[plot_idx].set_xlabel('Side')
    axes[plot_idx].set_ylabel('Count')
    axes[plot_idx].legend()
    plot_idx += 1

    # Volatility distribution
    axes[plot_idx].hist(train_df['today_volatility'], bins=50, alpha=0.5, label='Train')
    if test_df is not None:
        axes[plot_idx].hist(test_df['today_volatility'], bins=50, alpha=0.5, label='Test')
    axes[plot_idx].set_title('Daily Volatility Distribution')
    axes[plot_idx].set_xlabel('Daily Volatility')
    axes[plot_idx].set_ylabel('Frequency')
    axes[plot_idx].legend()
    plot_idx += 1

    # Intra-order return distribution
    axes[plot_idx].hist(train_df['intra_order_return'], bins=50, alpha=0.5, label='Train')
    if test_df is not None:
        axes[plot_idx].hist(test_df['intra_order_return'], bins=50, alpha=0.5, label='Test')
    axes[plot_idx].set_title('Intra-order Return Distribution')
    axes[plot_idx].set_xlabel('Return')
    axes[plot_idx].set_ylabel('Frequency')
    axes[plot_idx].legend()
    plot_idx += 1

    plt.tight_layout()
    plt.show()


def create_execution_summary_table(train_orders, test_orders=None, trim=0):
    """
    Creates a summary table of key execution metrics for train and test orders.
    @param train_orders: List of order information dictionaries for training data
    @param test_orders: Optional list of order information dictionaries for test data
    @return: None (prints the table)
    """
    def calculate_metrics(orders):
        # Initialize metrics
        total_notional = 0
        weighted_slippage = 0
        weighted_duration = 0
        weighted_return = 0
        rewards = []
        action_percentages = []
        slippages = []
        durations = []
        returns = []
        
        for order_info in orders:
            # Get the first and last step info
            first_step = order_info[0]
            last_step = order_info[-1]
            
            # Calculate order notional value assuming all currency is the same (USD in this case)
            notional = first_step['order_qty'] * last_step['order_vwap']
            total_notional += notional
            
            # Calculate total reward for the order
            total_reward = last_step.get('total_reward', 0)
            rewards.append(total_reward)
            
            # Calculate slippage (VWAP vs arrival price)
            slippage = (last_step['order_vwap'] - first_step['arrival_price']) / first_step['arrival_price']
            weighted_slippage += slippage * notional
            # side adjustment for slippage direction
            if first_step['side'] == 'sell':
                slippage *= -1
            slippages.append(slippage)
            
            # Calculate completion duration ratio  
            # first time when shares_remaining is 0
            completion_time = next((step['current_step'] for step in order_info
                                  if step.get('shares_remaining', None) == 0),
                                 last_step['current_step'])
            if completion_time is None:
                completion_time = last_step['current_step']
            total_horizon = first_step['time_horizon']
            duration_ratio = completion_time / total_horizon
            weighted_duration += duration_ratio * notional
            durations.append(duration_ratio)
            
            # Calculate intra-order return
            # find last fill price in order  which is last_order_price > 0
            intra_return = (last_step['mid_price'] - first_step['arrival_price']) / first_step['arrival_price']
            weighted_return += intra_return * notional
            returns.append(intra_return)
            
            # Collect action percentages
            for step in order_info:
                if 'action_percentage' in step:
                    action_percentages.append(step['action_percentage'])
        
        # Normalize by total notional
        if total_notional > 0:
            weighted_slippage /= total_notional
            weighted_duration /= total_notional
            weighted_return /= total_notional
        
        # Calculate standard deviations
        slippage_std = np.std(slippages) * 10000 if slippages else 0  # Convert to bps
        duration_std = np.std(durations) if durations else 0
        return_std = np.std(returns) * 10000 if returns else 0  # Convert to bps
        action_std = np.std(action_percentages) * 100 if action_percentages else 0  # Convert to percentage
        reward_std = np.std(rewards) * 10000 if rewards else 0  # Convert to bps
        
        # Calculate mean action percentage
        mean_action = np.mean(action_percentages) * 100 if action_percentages else 0  # Convert to percentage
        mean_reward = np.mean(rewards) * 10000 if rewards else 0  # Convert to bps
        
        return {
            'Weighted Slippage (bps)': weighted_slippage * 10000,  # Convert to basis points
            'Slippage Std Dev (bps)': slippage_std,
            'Weighted Duration Ratio': weighted_duration,
            'Duration Std Dev': duration_std,
            'Weighted Intra-Order Return (bps)': weighted_return * 10000,  # Convert to basis points
            'Return Std Dev (bps)': return_std,
            'Mean Reward (bps)': mean_reward,  # Already in bps
            'Reward Std Dev (bps)': reward_std,
            'Mean Action %': mean_action,
            'Action % Std Dev': action_std
        }
    
    # Calculate metrics for train and test
    train_metrics = calculate_metrics(train_orders)
    
    # Create the table
    print("\nExecution Summary Table")
    print("=" * 100)
    # Print number of orders
    print(f"Number of Orders - Train: {len(train_orders)}, Test: {len(test_orders) if test_orders else 'N/A'}")
    print("-" * 100)
    print(f"{'Metric':<30} {'Train':>15} {'Train Std':>15} {'Test':>15} {'Test Std':>15}")
    print("-" * 100)
    
    # Define the metrics to display in order
    metrics = [
        ('Weighted Slippage (bps)', 'Slippage Std Dev (bps)'),
        ('Weighted Duration Ratio', 'Duration Std Dev'),
        ('Weighted Intra-Order Return (bps)', 'Return Std Dev (bps)'),
        ('Mean Reward (bps)', 'Reward Std Dev (bps)'),
        ('Mean Action %', 'Action % Std Dev')
    ]
    
    # Calculate test metrics if available
    test_metrics = calculate_metrics(test_orders) if test_orders else None
    
    # Display each metric pair
    for metric, std_metric in metrics:
        train_value = f"{train_metrics[metric]:.2f}"
        train_std = f"{train_metrics[std_metric]:.2f}"
        
        if test_metrics:
            test_value = f"{test_metrics[metric]:.2f}"
            test_std = f"{test_metrics[std_metric]:.2f}"
        else:
            test_value = "N/A"
            test_std = "N/A"
            
        print(f"{metric:<30} {train_value:>15} {train_std:>15} {test_value:>15} {test_std:>15}")
    
    print("=" * 100)
    print("Note: Slippage and Returns are in basis points (bps)")
    print("      Duration Ratio is completion time / total horizon")
    print("      Action percentages are in %")
