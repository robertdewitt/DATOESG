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
    Args:
        orders_dict: Dictionary where keys are model names and values are lists of order information dictionaries.
        num_orders: Number of orders to plot per model.
    Returns:
        None (prints the plot)
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
    Args:
        orders_df: DataFrame containing order information.
        num_orders: Number of orders to display.
        name: Name of the dataset.
    Returns:
        None (prints the table)
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
    Args:
        a_df: DataFrame containing order information for the first dataset.
        b_df: DataFrame containing order information for the second dataset (optional).
        date_col: Name of the date column in the dataframes.
        a_name: Name of the first dataset.
        b_name: Name of the second dataset.
    Returns:
        None (prints the plots)
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
    Args:
        orders_dict: Dictionary where keys are model/dataset names and values are lists of order information dictionaries.
        trim: Trimming parameter for slippage (e.g., 0.01 for 1% top and bottom trim)
    Returns:
        None (prints the table)
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

            # Compute order notional (for weighting and logging)
            notional = first_step['order_qty'] * last_step['order_vwap']

            # Compute slippage and intra-order return
            arrival_price = first_step.get('arrival_price', 0)
            if arrival_price and arrival_price != 0:
                slippage = (arrival_price - last_step['order_vwap']) / arrival_price
                if first_step['side'] == 'sell':
                    slippage = -slippage
                raw_return = (last_step['mid_price'] - arrival_price) / arrival_price
                intra_return = raw_return if first_step.get('side', 'buy') == 'buy' else -raw_return
            else:
                slippage = 0.0
                intra_return = 0.0
                orders_with_no_arrival_price += 1

            # Sigma-based outlier filter (drop if either metric > 3 × sigma)
            # Determine sigma from analytics-provided daily_volatility; fall back to lag1
            dv_obj = first_step.get('daily_volatility', None)
            dv_val = None
            if dv_obj is not None:
                try:
                    dv_val = float(dv_obj)
                except Exception:
                    dv_val = None
            if (dv_val is None) or (not np.isfinite(dv_val)) or (dv_val <= 0):
                dv1_obj = first_step.get('daily_volatility_lag1', None)
                if dv1_obj is not None:
                    try:
                        dv_val = float(dv1_obj)
                    except Exception:
                        dv_val = None
            if dv_val is not None and dv_val > 0 and dv_val < 2.0:
                if (abs(slippage) > 3.0 * dv_val) or (abs(intra_return) > 3.0 * dv_val):
                    logger.warning(
                        "Dropping outlier order: ticker=%s date=%s side=%s qty=%s horizon=%s slippage_bps=%.1f intra_ret_bps=%.1f sigma=%.4f",
                        first_step.get('ticker', 'NA'),
                        first_step.get('date', 'NA'),
                        first_step.get('side', 'NA'),
                        first_step.get('order_qty', 'NA'),
                        first_step.get('time_horizon', 'NA'),
                        slippage * 10000.0,
                        intra_return * 10000.0,
                        dv_val,
                    )
                    continue

            else:
                logger.warning(
                    "Dropping order due to no daily volatility: ticker=%s date=%s side=%s qty=%s horizon=%s slippage_bps=%s intra_ret_bps=%s sigma=%s",
                    first_step.get('ticker', 'NA'),
                    first_step.get('date', 'NA'),
                    first_step.get('side', 'NA'),
                    first_step.get('order_qty', 'NA'),
                    first_step.get('time_horizon', 'NA'),
                    f"{slippage * 10000.0:.1f}",
                    f"{intra_return * 10000.0:.1f}",
                    "NA" if dv_val is None else f"{dv_val:.4f}",
                )

            # Accumulate only non-outliers
            total_notional += notional
            total_reward = last_step.get('total_reward', 0)
            rewards.append(total_reward)

            # Incomplete check (count only if kept)
            if last_step.get('shares_remaining', 0) > 0:
                incomplete_orders += 1

            weighted_return += intra_return * notional
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
            duration_ratio = completion_time / total_horizon if total_horizon > 0 else 1.0
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
        
        # Calculate standard errors (using all data, not trimmed)
        n = len(slippages)
        slippage_stderr = (np.std(slippages) / np.sqrt(n)) * 10000 if n > 0 else 0  # Convert to bps
        duration_stderr = (np.std(durations) / np.sqrt(n)) if n > 0 else 0
        return_stderr = (np.std(returns) / np.sqrt(n)) * 10000 if n > 0 else 0  # Convert to bps
        action_stderr = (np.std(action_percentages) / np.sqrt(len(action_percentages))) * 100 if action_percentages else 0  # Convert to percentage
        reward_stderr = (np.std(rewards) / np.sqrt(len(rewards))) * 10000 if rewards else 0  # Convert to bps
        
        # Calculate mean action percentage and reward
        mean_action = np.mean(action_percentages) * 100 if action_percentages else 0  # Convert to percentage
        mean_reward = np.mean(rewards) * 10000 if rewards else 0  # Convert to bps
        
        # Calculate percentage of incomplete orders
        incomplete_pct = (incomplete_orders / total_orders * 100) if total_orders > 0 else 0
        
        # orders with no arrival percentage 
        orders_with_no_arrival_pct = (orders_with_no_arrival_price / total_orders * 100) if total_orders > 0 else 0
        
        # Calculate trimmed count and notional if trimming was applied
        trimmed_count = 0
        trimmed_notional = 0
        if trim > 0 and len(slippage_notional_pairs) > 0:
            # Count how many orders were trimmed (not included in the middle portion)
            sorted_pairs = sorted(slippage_notional_pairs, key=lambda x: x[0])
            total_notional_for_trim = sum(pair[1] for pair in sorted_pairs)
            lower_trim_notional = total_notional_for_trim * trim
            upper_trim_notional = total_notional_for_trim * (1 - trim)
            
            cumulative_notional = 0
            for slippage, notional in sorted_pairs:
                cumulative_notional += notional
                if cumulative_notional < lower_trim_notional or cumulative_notional > upper_trim_notional:
                    trimmed_count += 1
                    trimmed_notional += notional

        return {
            'Count': total_orders,
            'Notional (M)': total_notional / 1e6,  # Convert to millions
            'Trimmed Count': trimmed_count if trim > 0 else None,
            'Trimmed Notional (M)': trimmed_notional / 1e6 if trim > 0 else None,  # Convert to millions
            'Weighted Slippage (bps)': weighted_slippage * 10000,  # Convert to basis points
            'Slippage Std Err (bps)': slippage_stderr,
            'Weighted Duration Ratio': weighted_duration,
            'Duration Std Err': duration_stderr,
            'Weighted Intra-Order Return (bps)': weighted_return * 10000,  # Convert to basis points
            'Return Std Err (bps)': return_stderr,
            'Mean Reward (bps)': mean_reward,  # Already in bps
            'Reward Std Err (bps)': reward_stderr,
            'Mean Action %': mean_action,
            'Action % Std Err': action_stderr,
            'Incomplete Orders %': incomplete_pct,
            'Orders with no arrival price %': orders_with_no_arrival_pct
        }
    
    # Calculate metrics for all models
    all_metrics = {}
    model_names = list(orders_dict.keys())
    
    for model_name, orders in orders_dict.items():
        all_metrics[model_name] = calculate_metrics(orders)
    
    # Define column headers (metrics)
    columns = [
        'Count',
        'Notional (M)',
    ]
    
    # Add trimmed columns if trimming is enabled
    if trim > 0:
        columns.extend(['Trim Count', 'Trim Not (M)'])
    
    columns.extend([
        'Slippage (bps)',
        'Slippage SE (bps)',
        'Duration Ratio', 
        'Duration SE',
        'Return (bps)',
        'Return SE (bps)',
        'Reward (bps)',
        'Reward SE (bps)',
        'Action %',
        'Action SE %',
        'Incomplete %',
        'No Arrival %'
    ])
    
    # Create column width calculation
    col_width = 12
    agent_col_width = 20
    
    # Create the table
    print("\nExecution Summary Table")
    print("=" * (agent_col_width + col_width * len(columns) + len(columns) - 1))
    
    if trim > 0:
        print(f"Slippage trimmed (top/bottom {trim:.1%} by notional weight)")
        print("-" * (agent_col_width + col_width * len(columns) + len(columns) - 1))
    
    # Create header row
    header = f"{'Agent':<{agent_col_width}}"
    for col in columns:
        header += f"{col:>{col_width}}|"
    print(header)
    print("-" * (agent_col_width + col_width * len(columns) + len(columns) - 1))
    
    # Map column names to metric keys
    column_to_metric = {
        'Count': 'Count',
        'Notional (M)': 'Notional (M)',
        'Trim Count': 'Trimmed Count',
        'Trim Not (M)': 'Trimmed Notional (M)',
        'Slippage (bps)': 'Weighted Slippage (bps)',
        'Slippage SE (bps)': 'Slippage Std Err (bps)',
        'Duration Ratio': 'Weighted Duration Ratio',
        'Duration SE': 'Duration Std Err',
        'Return (bps)': 'Weighted Intra-Order Return (bps)',
        'Return SE (bps)': 'Return Std Err (bps)',
        'Reward (bps)': 'Mean Reward (bps)',
        'Reward SE (bps)': 'Reward Std Err (bps)',
        'Action %': 'Mean Action %',
        'Action SE %': 'Action % Std Err',
        'Incomplete %': 'Incomplete Orders %',
        'No Arrival %': 'Orders with no arrival price %'
    }
    
    # Print each agent as a row - handle duplicate names
    model_name_counts = {}
    for model_name in model_names:
        model_metrics = all_metrics[model_name]
        
        # Truncate and handle duplicates
        truncated_name = model_name[:agent_col_width-2]
        if truncated_name in model_name_counts:
            model_name_counts[truncated_name] += 1
            display_name = f"{truncated_name}_{model_name_counts[truncated_name]}"
        else:
            model_name_counts[truncated_name] = 0
            display_name = truncated_name
        
        # Ensure display name fits in column width
        display_name = display_name[:agent_col_width]
        row = f"{display_name:<{agent_col_width}}"
        
        for col in columns:
            metric_key = column_to_metric[col]
            metric_value = model_metrics.get(metric_key)
            
            # Skip None values for trimmed columns when trim=0
            if metric_value is None:
                continue
                
            if col == 'Count':
                value = f"{int(metric_value):>{col_width}}"
            elif col == 'Trim Count':
                value = f"{int(metric_value):>{col_width}}"
            elif col in ['Notional (M)', 'Trim Not (M)']:
                value = f"{metric_value:>{col_width}.1f}"
            elif col in ['Duration Ratio', 'Duration SE']:
                value = f"{metric_value:>{col_width}.3f}"
            else:
                value = f"{metric_value:>{col_width}.2f}"
            row += value
        
        print(row)
    
    print("=" * (agent_col_width + col_width * len(columns) + len(columns) - 1))
    print("Note: Slippage, Returns, and Rewards are in basis points (bps)")
    print("      SE = Standard Error (std/√n)")
    print("      Duration Ratio is completion time / total horizon")
    print("      Action percentages are in %")
    print("      Incomplete % shows orders not fully executed")
    print("      No Arrival % shows orders without arrival price data")


def plot_arrival_slippage_by_factors(orders_dict, factors=None, bins=12, model_names=None):
    """
    Plots arrival price slippage (bps) with standard error bars for multiple models
    across several explanatory factors on the x-axis. Each model is a separate line.

    Args:
        orders_dict: Dict[str, list[list[dict]]]
            Keys are model names; values are lists of orders, where each order is a
            list of per-step dictionaries (as produced by the vectorized executor).
        factors: List[str] | None
            X-axes to plot. Supported: 'ehv_pct', 'time_horizon', 'daily_volatility', 'intra_order_return'.
            If None, defaults to the list above (will skip any missing columns gracefully).
        bins: int
            Number of bins for the x-axis when computing means/standard errors.
        model_names: Optional[List[str]]
            An explicit ordering of model names to plot. If None, uses orders_dict keys order.
    """
    if factors is None:
        factors = ['ehv_pct', 'time_horizon', 'daily_volatility', 'intra_order_return']

    def _orders_to_df(orders):
        rows = []
        for order_info in orders:
            # Guard for numpy arrays/Series: use explicit length/None checks
            if order_info is None:
                continue
            try:
                is_empty = (len(order_info) == 0)
            except Exception:
                is_empty = False
            if is_empty:
                continue
            first_step = order_info[0]
            last_step = order_info[-1]

            arrival_price = first_step.get('arrival_price', None)
            order_vwap = last_step.get('order_vwap', None)
            side = first_step.get('side', 'buy')

            slippage = None
            if arrival_price and order_vwap:
                slippage = (arrival_price - order_vwap) / arrival_price
                if side == 'sell':
                    slippage = -slippage

            intra_return = None
            if arrival_price and (last_step.get('mid_price', None) is not None):
                raw_return = (last_step['mid_price'] - arrival_price) / arrival_price
                # Side-adjust: positive means favorable move (up for buys, down for sells)
                intra_return = raw_return if side == 'buy' else -raw_return

            # Sigma-based outlier filter when daily_volatility available
            # Determine sigma for plotting filter from analytics; fall back to lag1
            dv_obj = first_step.get('daily_volatility', None)
            dv_val = None
            if dv_obj is not None:
                try:
                    dv_val = float(dv_obj)
                except Exception:
                    dv_val = None
            if (dv_val is None) or (not np.isfinite(dv_val)) or (dv_val <= 0):
                dv1_obj = first_step.get('daily_volatility_lag1', None)
                if dv1_obj is not None:
                    try:
                        dv_val = float(dv1_obj)
                    except Exception:
                        dv_val = None
            if (dv_val is not None and dv_val > 0 and dv_val < 2.0) and (slippage is not None or intra_return is not None):
                if ((slippage is not None and abs(slippage) > 3.0 * dv_val) or
                    (intra_return is not None and abs(intra_return) > 3.0 * dv_val)):
                    logger.warning(
                        "Dropping outlier order in plotting: ticker=%s date=%s side=%s slippage_bps=%s intra_ret_bps=%s sigma=%.4f",
                        first_step.get('ticker', 'NA'),
                        first_step.get('date', 'NA'),
                        first_step.get('side', 'NA'),
                        'NA' if slippage is None else f"{slippage * 10000.0:.1f}",
                        'NA' if intra_return is None else f"{intra_return * 10000.0:.1f}",
                        dv_val,
                    )
                    continue
            else:
                logger.warning(
                    "Dropping order due to no daily volatility: ticker=%s date=%s side=%s slippage_bps=%s intra_ret_bps=%s sigma=%s",
                    first_step.get('ticker', 'NA'),
                    first_step.get('date', 'NA'),
                    first_step.get('side', 'NA'),
                    'NA' if slippage is None else f"{slippage * 10000.0:.1f}",
                    'NA' if intra_return is None else f"{intra_return * 10000.0:.1f}",
                    "NA" if dv_val is None else f"{dv_val:.4f}",
                )

            rows.append({
                'slippage': slippage,
                'ehv_pct': first_step.get('ehv_pct', None),
                'time_horizon': first_step.get('time_horizon', None),
                'daily_volatility': first_step.get('daily_volatility', None),
                'intra_order_return': intra_return,
                'order_date': first_step.get('date', None),
            })

        if not rows:
            return pd.DataFrame(columns=['slippage'] + (factors or []))
        return pd.DataFrame(rows)

    model_names = model_names or list(orders_dict.keys())
    model_to_df = {name: _orders_to_df(orders_dict.get(name, [])) for name in model_names}

    # Build figure with a top-wide plot for date and 2x2 below for factors
    from matplotlib.dates import DateFormatter
    fig = plt.figure(figsize=(14, 14))
    gs = fig.add_gridspec(4, 2, height_ratios=[1.2, 0.8, 1.0, 1.0])
    ax_top = fig.add_subplot(gs[0, :])
    ax_box = fig.add_subplot(gs[1, :])
    axes = [
        fig.add_subplot(gs[2, 0]),
        fig.add_subplot(gs[2, 1]),
        fig.add_subplot(gs[3, 0]),
        fig.add_subplot(gs[3, 1]),
    ]

    # Color cycle for multiple models (define before any plotting uses it)
    color_cycle = plt.rcParams['axes.prop_cycle'].by_key().get('color', [
        'tab:blue', 'tab:orange', 'tab:green', 'tab:red', 'tab:purple',
        'tab:brown', 'tab:pink', 'tab:gray', 'tab:olive', 'tab:cyan'
    ])

    # Top plot: slippage by date (per model), with histogram counts on secondary axis
    # Gather all dates
    all_dates = []
    for df in model_to_df.values():
        if 'order_date' in df.columns and df['order_date'].notna().any():
            dates_series = pd.to_datetime(df['order_date'], errors='coerce').dropna().dt.date
            if not dates_series.empty:
                all_dates.append(dates_series.unique())
    # Flatten and sort unique dates
    if all_dates:
        unique_dates = sorted(set(np.concatenate(all_dates)))
        # Convert to week periods for aggregation and fewer ticks
        week_periods = pd.to_datetime(pd.Series(unique_dates)).dt.to_period('W').dt.start_time.dt.date
        unique_weeks = sorted(set(week_periods))
        x_positions = np.arange(len(unique_weeks), dtype=float)
        # Reasonable bar width per model
        num_models = max(1, len(model_names))
        per_model_width = 0.8 / num_models
        # Secondary axis for counts
        ax_top_count = ax_top.twinx()
        ax_top.set_title("Arrival Slippage by Date")
        ax_top.set_ylabel("Arrival Slippage (bps)")
        ax_top_count.set_ylabel("Order Count")

        for mi, model_name in enumerate(model_names):
            label_name = (model_name or "")[0:25]
            df = model_to_df[model_name]
            if df.empty or 'slippage' not in df.columns:
                continue
            valid = df['slippage'].notna() & df['order_date'].notna()
            if not valid.any():
                continue
            dfv = df.loc[valid, ['order_date', 'slippage']].copy()
            dfv['order_date'] = pd.to_datetime(dfv['order_date'], errors='coerce').dt.to_period('W').dt.start_time.dt.date
            dfv = dfv.dropna(subset=['order_date'])
            grouped = dfv.groupby('order_date')['slippage']
            means = grouped.mean() * 10000.0
            counts = grouped.count().astype(float)
            stds = grouped.std(ddof=1).fillna(0.0)
            with np.errstate(divide='ignore', invalid='ignore'):
                se = (stds / np.sqrt(counts)).fillna(0.0) * 10000.0

            # Align to all dates
            means_full = pd.Series(index=unique_weeks, dtype=float)
            means_full.loc[means.index] = means.values
            se_full = pd.Series(index=unique_weeks, dtype=float)
            se_full.loc[se.index] = se.values
            counts_full = pd.Series(0.0, index=unique_weeks)
            counts_full.loc[counts.index] = counts.values

            # Plot counts as bars
            offset = (mi - (num_models - 1) / 2.0) * per_model_width
            ax_top_count.bar(x_positions + offset, counts_full.values, width=per_model_width,
                             color=color_cycle[mi % len(color_cycle)], alpha=0.25)
            # Plot mean slippage with SE
            ax_top.errorbar(x_positions, means_full.values, yerr=se_full.values,
                            label=label_name, color=color_cycle[mi % len(color_cycle)],
                            marker='o', linestyle='-')

        # Format x-axis with dates
        ax_top.set_xticks(x_positions)
        ax_top.set_xticklabels([pd.to_datetime(d).strftime('%Y-%m-%d') for d in unique_weeks], rotation=45, ha='right')
        ax_top.grid(True, alpha=0.3)
        ax_top.legend(loc='best')
    else:
        ax_top.text(0.5, 0.5, 'No date data to plot', ha='center', va='center', transform=ax_top.transAxes)

    # Boxplot row: overall arrival slippage per model (bps)
    box_data = []
    box_labels = []
    for model_name in model_names:
        df = model_to_df[model_name]
        if df.empty or 'slippage' not in df.columns:
            continue
        vals = (df['slippage'].dropna().values * 10000.0)
        if vals.size == 0:
            continue
        box_data.append(vals)
        box_labels.append((model_name or "")[0:25])
    if box_data:
        bp = ax_box.boxplot(box_data, patch_artist=True, labels=box_labels)
        for i, patch in enumerate(bp['boxes']):
            patch.set_facecolor(color_cycle[i % len(color_cycle)])
            patch.set_alpha(0.25)
        ax_box.set_title("Arrival Slippage Distribution (bps) by Model")
        ax_box.set_ylabel("Arrival Slippage (bps)")
        ax_box.grid(True, alpha=0.3)
    else:
        ax_box.text(0.5, 0.5, 'No slippage data for boxplot', ha='center', va='center', transform=ax_box.transAxes)

    # Remaining 2x2 factors below
    n_plots = min(4, len(factors))

    for plot_idx, factor in enumerate(factors[:n_plots]):
        ax = axes[plot_idx]
        ax.set_title(f"Arrival Slippage vs {factor}")
        ax.set_ylabel("Arrival Slippage (bps)")
        # Secondary axis for order counts (histogram)
        ax_count = ax.twinx()
        ax_count.set_ylabel("Order Count")

        all_values = []
        for df in model_to_df.values():
            if factor in df.columns and df[factor].notna().any():
                all_values.append(df[factor].dropna().values)
        if not all_values:
            ax.text(0.5, 0.5, f"No data for {factor}", ha='center', va='center', transform=ax.transAxes)
            continue

        concat_vals = np.concatenate(all_values)
        if np.nanmin(concat_vals) == np.nanmax(concat_vals):
            bin_edges = np.linspace(concat_vals.min() - 0.5, concat_vals.max() + 0.5, bins + 1)
        else:
            bin_edges = np.histogram_bin_edges(concat_vals, bins=bins)

        # Pre-compute bin centers/width for consistent histogram placement
        all_bins = pd.IntervalIndex.from_breaks(bin_edges, closed='right')
        bin_centers = np.array([iv.mid for iv in all_bins], dtype=float)
        # Use 80% of average bin width (more visible than min), split across models
        bin_widths = np.diff(bin_edges)
        pos_widths = bin_widths[bin_widths > 0] if len(bin_widths) > 0 else np.array([1.0])
        avg_width = float(np.mean(pos_widths)) if pos_widths.size > 0 else 1.0
        base_bar_width = avg_width * 0.8
        num_models = max(1, len(model_names))
        per_model_width = base_bar_width / num_models

        # Improve layering so bars are visible behind lines
        try:
            ax_count.set_zorder(1)
            ax.set_zorder(2)
            ax.patch.set_alpha(0)
        except Exception:
            pass

        for mi, model_name in enumerate(model_names):
            label_name = (model_name or "")[0:25]
            df = model_to_df[model_name]
            if df.empty or 'slippage' not in df.columns:
                continue
            valid = df['slippage'].notna() & df[factor].notna()
            dfv = df.loc[valid, [factor, 'slippage']].copy()
            if dfv.empty:
                continue

            dfv['bin'] = pd.cut(dfv[factor], bins=bin_edges, include_lowest=True)
            grouped = dfv.groupby('bin', observed=True)['slippage']

            means = grouped.mean() * 10000.0
            counts = grouped.count().astype(float)
            stds = grouped.std(ddof=1).fillna(0.0)
            with np.errstate(divide='ignore', invalid='ignore'):
                se = (stds / np.sqrt(counts)).fillna(0.0) * 10000.0
            # Align counts to all bins for histogram bars
            # Ensure bin index alignment (closed='right' to match pd.cut default)
            counts_full = counts.reindex(all_bins, fill_value=0.0)
            # Fallback: if counts are all zeros (index mismatch), recompute via np.histogram
            if float(np.nansum(counts_full.values)) == 0.0 and dfv[factor].size > 0:
                hist_counts, _ = np.histogram(dfv[factor].values.astype(float), bins=bin_edges)
                counts_full = pd.Series(hist_counts.astype(float), index=all_bins)
            # Horizontal offset per model to avoid overlap
            offset = (mi - (num_models - 1) / 2.0) * per_model_width
            # Plot histogram bars on secondary axis
            ax_count.bar(
                bin_centers + offset,
                counts_full.values,
                width=per_model_width,
                color=color_cycle[mi % len(color_cycle)],
                alpha=0.6,
                edgecolor='none',
                zorder=1,
                label=None,
            )
            # Ensure count axis starts at zero for visibility
            ymin, ymax = ax_count.get_ylim()
            if ymin > 0:
                ax_count.set_ylim(bottom=0)

            ax.errorbar(
                pd.IntervalIndex(means.index).mid,
                means.values,
                yerr=se.values,
                label=label_name,
                color=color_cycle[mi % len(color_cycle)],
                marker='o',
                linestyle='-',
                zorder=2,
            )

        # Ensure bars are within view on shared x-axis
        ax.set_xlim(bin_edges[0], bin_edges[-1])
        ax.set_xlabel(factor)
        ax.grid(True, alpha=0.3)
        ax.legend(loc='best')

    total_slots = len(axes)
    for j in range(n_plots, total_slots):
        fig.delaxes(axes[j])

    plt.tight_layout()
    plt.show()


def create_execution_summary_and_plots(orders_dict, trim=0):
    """
    Print the execution summary table and then render the arrival slippage plots below it.
    """
    create_execution_summary_table(orders_dict, trim=trim)
    plot_arrival_slippage_by_factors(orders_dict)