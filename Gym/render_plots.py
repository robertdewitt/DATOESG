import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import logging
from matplotlib.ticker import MaxNLocator


# Set up logger for this module
logger = logging.getLogger(__name__)


# Consistent, colorblind-friendly palette
COLOR = {
    'a': 'tab:blue',                 # primary dataset
    'b': 'tab:purple',               # secondary dataset (changed from orange)
    'fill_price': 'tab:blue',
    'market_vwap_price': 'tab:green',
    'order_vwap_price': 'tab:purple',
    'reward': 'tab:red',
    'shares_remaining': 'tab:gray',  # changed from orange
    'trade_size': 'tab:cyan',
    'action_percentage': 'tab:olive',
    'accumulated_impact': 'tab:gray',
    'immediate_impact': 'tab:pink',
    # Cost components
    'arrival_cost': 'tab:blue',
    'vwap_cost': 'tab:green',
    'rate_penalty': 'tab:orange',
    'unfilled_cost': 'tab:brown',
    'holding_risk_cost': 'tab:purple',
    'total_step_cost': 'tab:red',
}


def plot_orders(orders_dict, num_orders=3):
    """
    Plots the order information including prices, quantities, and action percentages for multiple models.
    @param orders_dict: Dictionary where keys are model names and values are lists of order information dictionaries.
                       Can also accept a single list for backward compatibility.
    @param num_orders: Number of orders to plot per model.
    Each order dictionary should contain keys like 'ticker', 'side', 'time_horizon', 'order_qty',
    'current_step', 'mid_price', 'immediate_impact', 'action_percentage', 'accumulated_impact',
    'last_fill_price', 'vwap_price', 'order_vwap', 'total_reward', 'last_trade_size', and 'shares_remaining'.
    @return: None
    """ 
    # Loop through each model
    for model_name, orders in orders_dict.items():
        print(f"\n=== Plotting orders for {model_name} ===")
        
        # Loop through orders for this model
        for i, order_info in enumerate(orders[:num_orders]):
            # Extract meta data for this order
            ticker = order_info[0]["ticker"]
            side = order_info[0]["side"]
            time_horizon = order_info[0]["time_horizon"]
            adv_val = order_info[0].get("adv", None)
            order_date = order_info[0].get("date", None)

            if 'order_qty' in order_info[0]:
                total_shares = order_info[0]['order_qty']
            else:
                total_shares = (order_info[0].get('shares_remaining', 0) +
                              order_info[0].get('last_trade_size', 0))

            # Create figure with 4 subplots (add costs panel)
            fig, (ax1, ax2, ax3, ax4) = plt.subplots(4, 1, figsize=(12, 16), sharex=True)
            
            # Set title for the entire figure
            subtitle_parts = [
                f"{model_name}",
                f"[{ticker} | {side.capitalize()}] Order {i+1}",
                f"Total {total_shares:,.0f} Shares",
                f"Horizon: {time_horizon}"
            ]
            if adv_val is not None:
                subtitle_parts.append(f"ADV: {adv_val:,.0f}")
            fig.suptitle(" | ".join(subtitle_parts), fontsize=14)

            # Plot 1: Prices
            ax1.set_ylabel("Price", color=COLOR['fill_price'])
            times = [x["current_step"] for x in order_info]
            mid_prices = [x["mid_price"] for x in order_info]
            
            # Convert to bps and percentages
            immediate_impact = [x.get("immediate_impact", 0) * 10000 for x in order_info]
            action_percentages = [x.get("action_percentage", 0) * 100 for x in order_info]
            accumulated_impacts = [x.get("accumulated_impact", np.nan) * 10000 for x in order_info]

            # Filter out zero and invalid prices
            fill_prices = [x.get("last_fill_price", np.nan) if x.get("last_fill_price", 0) > 0 else np.nan for x in order_info]
            vwap_prices = [x.get("vwap_price", np.nan) if x.get("vwap_price", 0) > 0 else np.nan for x in order_info]
            order_vwap = [x.get("order_vwap", np.nan) if x.get("order_vwap", 0) > 0 else np.nan for x in order_info]
            total_rewards = [x.get("total_reward", np.nan) for x in order_info]

            # Only plot if we have valid prices
            if any(not np.isnan(p) for p in fill_prices):
                ax1.plot(times, fill_prices, label="Fill Price", color=COLOR['fill_price'])
            if any(not np.isnan(p) for p in vwap_prices):
                ax1.plot(times, vwap_prices, label="Market VWAP Price", color=COLOR['market_vwap_price'])
            if any(not np.isnan(p) for p in order_vwap):
                ax1.plot(times, order_vwap, label="Order VWAP Price", color=COLOR['order_vwap_price'], linestyle="--")
            ax1.grid(True)
           
            # Add reward on secondary y-axis
            ax1_reward = ax1.twinx()
            ax1_reward.set_ylabel("Total Reward", color=COLOR['reward'])
            ax1_reward.plot(times, total_rewards, label="Total Reward", color=COLOR['reward'], linestyle=":")
            
            # Combine legends
            lines1, labels1 = ax1.get_legend_handles_labels()
            lines2, labels2 = ax1_reward.get_legend_handles_labels()
            ax1.legend(lines1 + lines2, labels1 + labels2, loc="best")

            # Plot 2: Quantities
            ax2.set_ylabel("Quantity", color=COLOR['shares_remaining'])
            trade_sizes = [x["last_trade_size"] for x in order_info]
            shares_remaining = [x["shares_remaining"] for x in order_info]

            ax2.plot(times, shares_remaining, label="Shares Remaining", color=COLOR['shares_remaining'], linestyle="--")
            ax2.grid(True)
              
            # Add trade size on secondary y-axis
            ax2_trade = ax2.twinx()
            ax2_trade.set_ylabel("Trade Size", color=COLOR['trade_size'])
            ax2_trade.scatter(times, trade_sizes, label="Trade Size", color=COLOR['trade_size'], marker="o")
            
            # Combine legends
            lines1, labels1 = ax2.get_legend_handles_labels()
            lines2, labels2 = ax2_trade.get_legend_handles_labels()
            ax2.legend(lines1 + lines2, labels1 + labels2, loc="best")

            # Plot 3: Action Percentage
            ax3.set_ylabel("Action %", color=COLOR['action_percentage'])
            ax3.set_xlabel("Time (minutes)")
            action_percentages = [x.get("action_percentage", 0) * 100 for x in order_info]
            ax3.plot(times, action_percentages, label="Action %", color=COLOR['action_percentage'])
            ax3.grid(True)
            
            # Add accumulated impact on secondary y-axis
            ax3_impact = ax3.twinx()
            ax3_impact.set_ylabel("Accumulated Impact (bps)", color=COLOR['accumulated_impact'])
            ax3_impact.plot(times, accumulated_impacts, label="Accumulated Impact (bps)",
                           color=COLOR['accumulated_impact'], linestyle="--")
            ax3_impact.scatter(times, immediate_impact, label="Immediate Impact (bps)",
                              color=COLOR['immediate_impact'], marker="x")
            
            # Combine legends
            lines1, labels1 = ax3.get_legend_handles_labels()
            lines2, labels2 = ax3_impact.get_legend_handles_labels()
            ax3.legend(lines1 + lines2, labels1 + labels2, loc="best")

            # Plot 4: Costs panel
            ax4.set_ylabel("Cost")
            ax4.set_xlabel("Time (minutes)")
            cost_keys = [
                'arrival_cost', 'vwap_cost', 'rate_penalty',
                'unfilled_cost', 'holding_risk_cost', 'total_step_cost'
            ]
            for key in cost_keys:
                series = [x.get(key, np.nan) for x in order_info]
                if any(not np.isnan(v) for v in series):
                    ax4.plot(times, series, label=key, color=COLOR.get(key, None))
            ax4.grid(True)
            ax4.legend(loc="best")

            # Set x-axis limits to show full horizon
            ax1.set_xlim(0, time_horizon)
            
            # Add vertical line at order completion time
            if any((x.get("shares_remaining", None) == 0) for x in order_info):
                completion_time = next(x["current_step"] for x in order_info if x.get("shares_remaining", None) == 0)
                for ax in [ax1, ax2, ax3, ax4]:
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
            elif (key == 'adv_pct') or (key == 'ehv_pct'):
                print(f"{key}: {val:.2%}", end=' ')
            else:
                print(f"{key}: {val}", end=' ')
        print()
        i += 1
        if i > num_orders:
            break
    print("========================================")


def plot_order_histograms(a_df, b_df=None, date_col='Datetime', a_name='Train', b_name='Test'):
    """
    Plots histograms of order characteristics comparing train and test distributions.
    @param train_df: DataFrame containing training order information.
    @param test_df: Optional DataFrame containing test order information.
    @param date_col: Column name containing the date/time information.
    @return: None
    """
    fig, axes = plt.subplots(3, 3, figsize=(15, 15))
    axes = axes.flatten()

    # Plot histograms for each characteristic
    plot_idx = 0
    
    # Winsorize daily volatility to 1st–99th percentile for cleaner visuals
    if 'daily_volatility' in a_df.columns:
        a_df = a_df.copy()
        a_low, a_high = np.nanpercentile(a_df['daily_volatility'].dropna(), [1, 99])
        a_df['daily_volatility_trimmed'] = a_df['daily_volatility'].clip(lower=a_low, upper=a_high)
    if b_df is not None and 'daily_volatility' in b_df.columns:
        b_df = b_df.copy()
        b_low, b_high = np.nanpercentile(b_df['daily_volatility'].dropna(), [1, 99])
        b_df['daily_volatility_trimmed'] = b_df['daily_volatility'].clip(lower=b_low, upper=b_high)
    
    # Order size distribution
    axes[plot_idx].hist(a_df['order_qty'], bins=50, alpha=0.5, label=a_name, color=COLOR['a'])
    if b_df is not None:
        axes[plot_idx].hist(b_df['order_qty'], bins=50, alpha=0.5, label=b_name, color=COLOR['b'])
    axes[plot_idx].set_title('Order Size Distribution')
    axes[plot_idx].set_xlabel('Order Size')
    axes[plot_idx].set_ylabel('Frequency')
    axes[plot_idx].legend()
    plot_idx += 1

    # Time horizon distribution
    axes[plot_idx].hist(a_df['time_horizon'], bins=50, alpha=0.5, label=a_name, color=COLOR['a'])
    if b_df is not None:
        axes[plot_idx].hist(b_df['time_horizon'], bins=50, alpha=0.5, label=b_name, color=COLOR['b'])
    axes[plot_idx].set_title('Time Horizon Distribution')
    axes[plot_idx].set_xlabel('Time Horizon (minutes)')
    axes[plot_idx].set_ylabel('Frequency')
    axes[plot_idx].legend()
    plot_idx += 1

    # ADV percentage distribution
    axes[plot_idx].hist(a_df['adv_pct'], bins=50, alpha=0.5, label=a_name, color=COLOR['a'])
    if b_df is not None:
        axes[plot_idx].hist(b_df['adv_pct'], bins=50, alpha=0.5, label=b_name, color=COLOR['b'])
    axes[plot_idx].set_title('ADV Percentage Distribution')
    axes[plot_idx].set_xlabel('ADV Percentage')
    axes[plot_idx].set_ylabel('Frequency')
    axes[plot_idx].legend()
    # Rotate x-axis labels for ADV percentage
    axes[plot_idx].tick_params(axis='x', labelrotation=45)
    plot_idx += 1

    # Daily Volatility by Month (box plots)
    a_df['date'] = pd.to_datetime(a_df['date'])
    a_df['month'] = a_df['date'].dt.to_period('M').dt.to_timestamp()
    train_daily_vol = a_df.groupby('month')[
        'daily_volatility_trimmed' if 'daily_volatility_trimmed' in a_df.columns else 'daily_volatility'
    ].apply(list).to_dict()

    if b_df is not None:
        b_df['date'] = pd.to_datetime(b_df['date'])
        b_df['month'] = b_df['date'].dt.to_period('M').dt.to_timestamp()
        test_daily_vol = b_df.groupby('month')[
            'daily_volatility_trimmed' if 'daily_volatility_trimmed' in b_df.columns else 'daily_volatility'
        ].apply(list).to_dict()

    all_months = sorted(set(train_daily_vol.keys()) | (set(test_daily_vol.keys()) if b_df is not None else set()))

    volatility_data = []
    colors = []
    positions = []

    for i, month in enumerate(all_months):
        if month in train_daily_vol:
            volatility_data.append(train_daily_vol[month])
            colors.append(COLOR['a'])
            positions.append(i)
        if b_df is not None and month in test_daily_vol:
            volatility_data.append(test_daily_vol[month])
            colors.append(COLOR['b'])
            positions.append(i + 0.35)

    bp1 = axes[plot_idx].boxplot(volatility_data, positions=positions, widths=0.3, patch_artist=True)

    for patch, color in zip(bp1['boxes'], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.5)

    axes[plot_idx].set_title('Daily Volatility by Month')
    axes[plot_idx].set_xlabel('Month')
    axes[plot_idx].set_ylabel('Daily Volatility')
    axes[plot_idx].set_xticks(range(len(all_months)))
    axes[plot_idx].set_xticklabels([pd.to_datetime(m).strftime('%Y-%m') for m in all_months], rotation=45, ha='right')

    from matplotlib.patches import Patch
    if b_df is None:
        legend_elements = [Patch(facecolor=COLOR['a'], alpha=0.5, label=a_name)]
    else:
        legend_elements = [Patch(facecolor=COLOR['a'], alpha=0.5, label=a_name),
                           Patch(facecolor=COLOR['b'], alpha=0.5, label=b_name)]
    axes[plot_idx].legend(handles=legend_elements)
    plot_idx += 1

    # Daily Volatility distribution (now after Monthly Box Plot; uses trimmed values if available)
    if 'daily_volatility_trimmed' in a_df.columns:
        axes[plot_idx].hist(a_df['daily_volatility_trimmed'], bins=50, alpha=0.5, label=a_name, color=COLOR['a'])
    else:
        axes[plot_idx].hist(a_df['daily_volatility'], bins=50, alpha=0.5, label=a_name, color=COLOR['a'])
    if b_df is not None:
        if hasattr(b_df, 'columns') and 'daily_volatility_trimmed' in b_df.columns:
            axes[plot_idx].hist(b_df['daily_volatility_trimmed'], bins=50, alpha=0.5, label=b_name, color=COLOR['b'])
        elif hasattr(b_df, 'columns') and 'daily_volatility' in b_df.columns:
            axes[plot_idx].hist(b_df['daily_volatility'], bins=50, alpha=0.5, label=b_name, color=COLOR['b'])
    axes[plot_idx].set_title('Daily Volatility Distribution')
    axes[plot_idx].set_xlabel('Daily Volatility')
    axes[plot_idx].set_ylabel('Frequency')
    axes[plot_idx].legend()
    plot_idx += 1

    # Intra-order return distribution
    axes[plot_idx].hist(a_df['intra_order_return'], bins=50, alpha=0.5, label=a_name, color=COLOR['a'])
    if b_df is not None:
        axes[plot_idx].hist(b_df['intra_order_return'], bins=50, alpha=0.5, label=b_name, color=COLOR['b'])
    axes[plot_idx].set_title('Intra-order Return Distribution')
    axes[plot_idx].set_xlabel('Return')
    axes[plot_idx].set_ylabel('Frequency')
    axes[plot_idx].legend()
    plot_idx += 1

    # Y values distribution (impact coefficient)
    if 'Y' in a_df.columns:
        y_data_a = a_df['Y'].dropna()
        y_data_b = b_df['Y'].dropna() if b_df is not None and 'Y' in b_df.columns else pd.Series()
        
        if len(y_data_a) > 0 or len(y_data_b) > 0:
            if len(y_data_a) > 0:
                axes[plot_idx].hist(y_data_a, bins=50, alpha=0.5, label=a_name, color=COLOR['a'])
            if len(y_data_b) > 0:
                axes[plot_idx].hist(y_data_b, bins=50, alpha=0.5, label=b_name, color=COLOR['b'])
            axes[plot_idx].set_title('Y Values Distribution (Impact Coefficient)')
            axes[plot_idx].set_xlabel('Y Value')
            axes[plot_idx].set_ylabel('Frequency')
            axes[plot_idx].legend()
            axes[plot_idx].tick_params(axis='x', labelrotation=45)
        else:
            axes[plot_idx].text(0.5, 0.5, 'No valid Y values to plot', ha='center', va='center', transform=axes[plot_idx].transAxes)
            axes[plot_idx].set_title('Y Values Distribution (No Data)')
    else:
        axes[plot_idx].text(0.5, 0.5, 'Y column not found in data', ha='center', va='center', transform=axes[plot_idx].transAxes)
        axes[plot_idx].set_title('Y Values Distribution (Column Not Found)')
    plot_idx += 1

    # Tau values distribution (decay rate)
    if 'tau' in a_df.columns:
        tau_data_a = a_df['tau'].dropna()
        tau_data_b = b_df['tau'].dropna() if b_df is not None and 'tau' in b_df.columns else pd.Series()
        
        if len(tau_data_a) > 0 or len(tau_data_b) > 0:
            if len(tau_data_a) > 0:
                axes[plot_idx].hist(tau_data_a, bins=50, alpha=0.5, label=a_name, color=COLOR['a'])
            if len(tau_data_b) > 0:
                axes[plot_idx].hist(tau_data_b, bins=50, alpha=0.5, label=b_name, color=COLOR['b'])
            axes[plot_idx].set_title('Tau Values Distribution (Decay Rate)')
            axes[plot_idx].set_xlabel('Tau Value')
            axes[plot_idx].set_ylabel('Frequency')
            axes[plot_idx].legend()
        else:
            axes[plot_idx].text(0.5, 0.5, 'No valid tau values to plot', ha='center', va='center', transform=axes[plot_idx].transAxes)
            axes[plot_idx].set_title('Tau Values Distribution (No Data)')
    else:
        axes[plot_idx].text(0.5, 0.5, 'Tau column not found in data', ha='center', va='center', transform=axes[plot_idx].transAxes)
        axes[plot_idx].set_title('Tau Values Distribution (Column Not Found)')
    plot_idx += 1

    # ADV Percentage Line Plot with Standard Error Bands
    # Prepare data for line plots
    train_daily_adv_pct = a_df.groupby(a_df['date'].dt.date)['adv_pct'].agg(['mean', 'std', 'count']).to_dict('index')
    
    if b_df is not None:
        test_daily_adv_pct = b_df.groupby(b_df['date'].dt.date)['adv_pct'].agg(['mean', 'std', 'count']).to_dict('index')
    
    # Prepare data for line plot
    train_dates = []
    train_means = []
    train_errors = []
    
    test_dates = []
    test_means = []
    test_errors = []
    
    # Build the unified date set for ADV plot (daily granularity)
    all_adv_dates = set(train_daily_adv_pct.keys())
    if b_df is not None:
        all_adv_dates |= set(test_daily_adv_pct.keys())
    for date in sorted(all_adv_dates):
        if date in train_daily_adv_pct:
            train_dates.append(date)
            mean_val = train_daily_adv_pct[date]['mean'] * 100  # Convert to percentage
            std_val = train_daily_adv_pct[date]['std'] * 100 if train_daily_adv_pct[date]['std'] is not None else 0
            count_val = train_daily_adv_pct[date]['count']
            std_error = std_val / np.sqrt(count_val) if count_val > 0 else 0
            
            train_means.append(mean_val)
            train_errors.append(std_error)
        
        if b_df is not None and date in test_daily_adv_pct:
            test_dates.append(date)
            mean_val = test_daily_adv_pct[date]['mean'] * 100  # Convert to percentage
            std_val = test_daily_adv_pct[date]['std'] * 100 if test_daily_adv_pct[date]['std'] is not None else 0
            count_val = test_daily_adv_pct[date]['count']
            std_error = std_val / np.sqrt(count_val) if count_val > 0 else 0
            
            test_means.append(mean_val)
            test_errors.append(std_error)
    
    # Create line plot with actual dates on x-axis
    if train_dates:
        x_train = pd.to_datetime(train_dates)
        axes[plot_idx].plot(x_train, train_means, 'o-', color=COLOR['a'], label=a_name, linewidth=2, markersize=4)
        axes[plot_idx].fill_between(x_train,
                                   [m - e for m, e in zip(train_means, train_errors)],
                                   [m + e for m, e in zip(train_means, train_errors)],
                                   color=COLOR['a'], alpha=0.2)
    
    if b_df is not None and test_dates:
        x_test = pd.to_datetime(test_dates)
        axes[plot_idx].plot(x_test, test_means, 's-', color=COLOR['b'], label=b_name, linewidth=2, markersize=4)
        axes[plot_idx].fill_between(x_test,
                                   [m - e for m, e in zip(test_means, test_errors)],
                                   [m + e for m, e in zip(test_means, test_errors)],
                                   color=COLOR['b'], alpha=0.2)
    
    axes[plot_idx].set_title('ADV Percentage by Date (Mean ± Standard Error)')
    axes[plot_idx].set_xlabel('Date')
    axes[plot_idx].set_ylabel('ADV Percentage (%)')
    axes[plot_idx].grid(True, alpha=0.3)
    # Rotate x-axis labels for ADV percentage by date
    axes[plot_idx].tick_params(axis='x', labelrotation=45)
    
    # Add legend
    axes[plot_idx].legend()
    plot_idx += 1

    date_axes_indices = [
        # these are the subplot indices where you're doing the date boxplots:
        # 6: Daily Volatility by Date
        # 7: Intra-order Returns by Date
        # 8: ADV Percentage by Date
        6, 7, 8
    ]

    for idx in date_axes_indices:
        ax = axes[idx]
        # ensure no more than 12 major ticks
        ax.xaxis.set_major_locator(MaxNLocator(nbins=12, integer=True))
 

    plt.tight_layout()
    plt.show()


def create_execution_summary_table(orders_dict, trim=0):
    """
    Creates a summary table of key execution metrics for multiple models/datasets.
    @param orders_dict: Dictionary where keys are model/dataset names and values are lists of order information dictionaries.
    @param trim: Trimming parameter for slippage (e.g., 0.01 for 1% top and bottom trim)
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
        incomplete_orders = 0
        orders_with_no_arrival_price = 0
        total_orders = len(orders)
        
        # For trimming, we need to collect slippages with their notional weights
        slippage_notional_pairs = []
        
        for order_info in orders:
            # Get the first and last step info
            first_step = order_info[0]
            last_step = order_info[-1]
            
            # Check if order is incomplete (shares_remaining > 0)
            if last_step.get('shares_remaining', 0) > 0:
                incomplete_orders += 1
            
            # Calculate order notional value assuming all currency is the same (USD in this case)
            notional = first_step['order_qty'] * last_step['order_vwap']
            total_notional += notional
            
            # Calculate total reward for the order
            total_reward = last_step.get('total_reward', 0)
            rewards.append(total_reward)
            
            # Calculate slippage (VWAP vs arrival price)
            arrival_price = first_step.get('arrival_price', 0)
            if arrival_price and arrival_price != 0:
                slippage = (arrival_price - last_step['order_vwap']) / arrival_price
                # side adjustment for slippage direction
                if first_step['side'] == 'sell':
                    slippage = -slippage
                            # Calculate intra-order return
                intra_return = (last_step['mid_price'] - arrival_price) / arrival_price
                weighted_return += intra_return * notional
            
            else:
                # Undefined arrival price – set slippage to 0 and warn once
                slippage = 0.0
                intra_return = 0.0
                orders_with_no_arrival_price += 1

            returns.append(intra_return)
            slippages.append(slippage)
            slippage_notional_pairs.append((slippage, notional))
            
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
            
            
            # Collect action percentages
            for step in order_info:
                if 'action_percentage' in step:
                    action_percentages.append(step['action_percentage'])
        
        # Apply trimming to slippage calculation
        if trim > 0 and len(slippage_notional_pairs) > 0:
            # Sort by slippage values
            sorted_pairs = sorted(slippage_notional_pairs, key=lambda x: x[0])
            
            # Calculate cumulative notional weights
            total_notional_for_trim = sum(pair[1] for pair in sorted_pairs)
            
            # Find trim boundaries based on notional weights
            lower_trim_notional = total_notional_for_trim * trim
            upper_trim_notional = total_notional_for_trim * (1 - trim)
            
            cumulative_notional = 0
            trimmed_pairs = []
            
            for slippage, notional in sorted_pairs:
                cumulative_notional += notional
                # Keep only the middle portion (exclude top and bottom trim percentiles)
                if lower_trim_notional <= cumulative_notional <= upper_trim_notional:
                    trimmed_pairs.append((slippage, notional))
            
            # Calculate weighted slippage from trimmed data
            if trimmed_pairs:
                trimmed_total_notional = sum(pair[1] for pair in trimmed_pairs)
                weighted_slippage = sum(slippage * notional for slippage, notional in trimmed_pairs) / trimmed_total_notional
            else:
                weighted_slippage = 0
        else:
            # No trimming - use all data
            weighted_slippage = sum(slippage * notional for slippage, notional in slippage_notional_pairs) / total_notional if total_notional > 0 else 0
        
        # Normalize other metrics by total notional (using original total_notional, not trimmed)
        if total_notional > 0:
            weighted_duration /= total_notional
            weighted_return /= total_notional
        
        # Calculate standard deviations (using all data, not trimmed)
        slippage_std = np.std(slippages) * 10000 if slippages else 0  # Convert to bps
        duration_std = np.std(durations) if durations else 0
        return_std = np.std(returns) * 10000 if returns else 0  # Convert to bps
        action_std = np.std(action_percentages) * 100 if action_percentages else 0  # Convert to percentage
        reward_std = np.std(rewards) * 10000 if rewards else 0  # Convert to bps
        
        # Calculate mean action percentage
        mean_action = np.mean(action_percentages) * 100 if action_percentages else 0  # Convert to percentage
        mean_reward = np.mean(rewards) * 10000 if rewards else 0  # Convert to bps
        
        # Calculate percentage of incomplete orders
        incomplete_pct = (incomplete_orders / total_orders * 100) if total_orders > 0 else 0
        
        # orders with no arrival perventage 
        orders_with_no_arrival_pct = (orders_with_no_arrival_price / total_orders * 100) if total_orders > 0 else 0

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
            'Action % Std Dev': action_std,
            'Incomplete Orders %': incomplete_pct,
            'Orders with no arrival price %': orders_with_no_arrival_pct
        }
    
    # Calculate metrics for all models
    all_metrics = {}
    model_names = list(orders_dict.keys())
    
    for model_name, orders in orders_dict.items():
        all_metrics[model_name] = calculate_metrics(orders)
    
    # Create the table
    print("\nExecution Summary Table")
    print("=" * (30 + 16 * len(model_names) * 2))
    
    # Print number of orders
    order_counts = " | ".join([f"{model}: {len(orders)}" for model, orders in orders_dict.items()])
    print(f"Number of Orders - {order_counts}")
    if trim > 0:
        print(f"Slippage trimmed (top/bottom {trim:.1%} by notional weight)")
    print("-" * (30 + 16 * len(model_names) * 2))
    
    # Create header with model names
    header = f"{'Metric':<30}"
    for model_name in model_names:
        header += f"{model_name:>15} {model_name + ' Std':>15}"
    print(header)
    print("-" * (30 + 16 * len(model_names) * 2))
    
    # Define the metrics to display in order
    metrics = [
        ('Weighted Slippage (bps)', 'Slippage Std Dev (bps)'),
        ('Weighted Duration Ratio', 'Duration Std Dev'),
        ('Weighted Intra-Order Return (bps)', 'Return Std Dev (bps)'),
        ('Mean Reward (bps)', 'Reward Std Dev (bps)'),
        ('Mean Action %', 'Action % Std Dev'),
        ('Incomplete Orders %', None),
        ('Orders with no arrival price %', None)  # No std dev for percentage
    ]
    
    # Display each metric pair
    for metric, std_metric in metrics:
        row = f"{metric:<30}"
        
        for model_name in model_names:
            model_metrics = all_metrics[model_name]
            value = f"{model_metrics[metric]:.2f}"
            if std_metric and std_metric in model_metrics:
                std_value = f"{model_metrics[std_metric]:.2f}"
                row += f"{value:>15} {std_value:>15}"
            else:
                # For metrics without std dev (like Incomplete Orders %), just show the value twice or add spacing
                row += f"{value:>15} {'-':>15}"
            
        print(row)
    
    print("=" * (30 + 16 * len(model_names) * 2))
    print("Note: Slippage and Returns are in basis points (bps)")
    print("      Duration Ratio is completion time / total horizon")
    print("      Action percentages are in %")
    print("      Incomplete Orders % shows orders not fully executed")
