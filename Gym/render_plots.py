import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import logging
from matplotlib.ticker import MaxNLocator


# Set up logger for this module
logger = logging.getLogger(__name__)
# set log level to warning
logger.setLevel(logging.WARNING)

# Consistent, colorblind-friendly palette
COLOR = {
    'a': 'tab:blue',                 # primary dataset (train)
    'b': 'tab:orange',               # secondary dataset (validation)
    'c': 'tab:purple',               # tertiary dataset (test)
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


def plot_order_histograms(a_df, b_df=None, c_df=None, date_col='Datetime', a_name='Train', b_name='Validation', c_name='Test'):
    """
    Plots histograms of order characteristics comparing train, validation, and test distributions.
    Args:
        a_df: DataFrame containing order information for the first dataset (train).
        b_df: DataFrame containing order information for the second dataset (validation, optional).
        c_df: DataFrame containing order information for the third dataset (test, optional).
        date_col: Name of the date column in the dataframes.
        a_name: Name of the first dataset.
        b_name: Name of the second dataset.
        c_name: Name of the third dataset.
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
    if c_df is not None and 'daily_volatility' in c_df.columns:
        c_df = c_df.copy()
        c_low, c_high = np.nanpercentile(c_df['daily_volatility'].dropna(), [1, 99])
        c_df['daily_volatility_trimmed'] = c_df['daily_volatility'].clip(lower=c_low, upper=c_high)
    
    # Order size distribution
    axes[plot_idx].hist(a_df['order_qty'], bins=50, alpha=0.5, label=a_name, color=COLOR['a'])
    if b_df is not None:
        axes[plot_idx].hist(b_df['order_qty'], bins=50, alpha=0.5, label=b_name, color=COLOR['b'])
    if c_df is not None:
        axes[plot_idx].hist(c_df['order_qty'], bins=50, alpha=0.5, label=c_name, color=COLOR['c'])
    axes[plot_idx].set_title('Order Size Distribution')
    axes[plot_idx].set_xlabel('Order Size')
    axes[plot_idx].set_ylabel('Frequency')
    axes[plot_idx].legend()
    plot_idx += 1

    # Time horizon distribution
    axes[plot_idx].hist(a_df['time_horizon'], bins=50, alpha=0.5, label=a_name, color=COLOR['a'])
    if b_df is not None:
        axes[plot_idx].hist(b_df['time_horizon'], bins=50, alpha=0.5, label=b_name, color=COLOR['b'])
    if c_df is not None:
        axes[plot_idx].hist(c_df['time_horizon'], bins=50, alpha=0.3, label=c_name, color=COLOR['c'])
    axes[plot_idx].set_title('Time Horizon Distribution')
    axes[plot_idx].set_xlabel('Time Horizon (minutes)')
    axes[plot_idx].set_ylabel('Frequency')
    axes[plot_idx].legend()
    plot_idx += 1

    # ADV percentage distribution
    axes[plot_idx].hist(a_df['adv_pct'], bins=50, alpha=0.5, label=a_name, color=COLOR['a'])
    if b_df is not None:
        axes[plot_idx].hist(b_df['adv_pct'], bins=50, alpha=0.5, label=b_name, color=COLOR['b'])
    if c_df is not None:
        axes[plot_idx].hist(c_df['adv_pct'], bins=50, alpha=0.3, label=c_name, color=COLOR['c'])
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

    if c_df is not None:
        c_df['date'] = pd.to_datetime(c_df['date'])
        c_df['month'] = c_df['date'].dt.to_period('M').dt.to_timestamp()
        test2_daily_vol = c_df.groupby('month')[
            'daily_volatility_trimmed' if 'daily_volatility_trimmed' in c_df.columns else 'daily_volatility'
        ].apply(list).to_dict()

    all_months = sorted(set(train_daily_vol.keys()) | (set(test_daily_vol.keys()) if b_df is not None else set()) | (set(test2_daily_vol.keys()) if c_df is not None else set()))

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

        if c_df is not None and month in test2_daily_vol:
            volatility_data.append(test2_daily_vol[month])
            colors.append(COLOR['c'])
            positions.append(i + 0.70)

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
    if c_df is not None:
        if hasattr(c_df, 'columns') and 'daily_volatility_trimmed' in c_df.columns:
            axes[plot_idx].hist(c_df['daily_volatility_trimmed'], bins=50, alpha=0.5, label=c_name, color=COLOR['c'])
        elif hasattr(c_df, 'columns') and 'daily_volatility' in c_df.columns:
            axes[plot_idx].hist(c_df['daily_volatility'], bins=50, alpha=0.5, label=c_name, color=COLOR['c'])
    axes[plot_idx].set_title('Daily Volatility Distribution')
    axes[plot_idx].set_xlabel('Daily Volatility')
    axes[plot_idx].set_ylabel('Frequency')
    axes[plot_idx].legend()
    plot_idx += 1

    # Intra-order return distribution
    axes[plot_idx].hist(a_df['intra_order_return'], bins=50, alpha=0.5, label=a_name, color=COLOR['a'])
    if b_df is not None:
        axes[plot_idx].hist(b_df['intra_order_return'], bins=50, alpha=0.5, label=b_name, color=COLOR['b'])
    if c_df is not None:
        axes[plot_idx].hist(c_df['intra_order_return'], bins=50, alpha=0.5, label=c_name, color=COLOR['c'])
    axes[plot_idx].set_title('Intra-order Return Distribution')
    axes[plot_idx].set_xlabel('Return')
    axes[plot_idx].set_ylabel('Frequency')
    axes[plot_idx].legend()
    plot_idx += 1

    # Y values distribution (impact coefficient)
    if 'Y' in a_df.columns:
        y_data_a = a_df['Y'].dropna()
        y_data_b = b_df['Y'].dropna() if b_df is not None and 'Y' in b_df.columns else pd.Series()
        y_data_c = c_df['Y'].dropna() if c_df is not None and 'Y' in c_df.columns else pd.Series()
        
        if len(y_data_a) > 0 or len(y_data_b) > 0 or len(y_data_c) > 0:
            if len(y_data_a) > 0:
                axes[plot_idx].hist(y_data_a, bins=50, alpha=0.5, label=a_name, color=COLOR['a'])
            if len(y_data_b) > 0:
                axes[plot_idx].hist(y_data_b, bins=50, alpha=0.5, label=b_name, color=COLOR['b'])
            if len(y_data_c) > 0:
                axes[plot_idx].hist(y_data_c, bins=50, alpha=0.5, label=c_name, color=COLOR['c'])
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
        tau_data_c = c_df['tau'].dropna() if c_df is not None and 'tau' in c_df.columns else pd.Series()
        
        if len(tau_data_a) > 0 or len(tau_data_b) > 0 or len(tau_data_c) > 0:
            if len(tau_data_a) > 0:
                axes[plot_idx].hist(tau_data_a, bins=50, alpha=0.5, label=a_name, color=COLOR['a'])
            if len(tau_data_b) > 0:
                axes[plot_idx].hist(tau_data_b, bins=50, alpha=0.5, label=b_name, color=COLOR['b'])
            if len(tau_data_c) > 0:
                axes[plot_idx].hist(tau_data_c, bins=50, alpha=0.5, label=c_name, color=COLOR['c'])
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
    
    if c_df is not None:
        test2_daily_adv_pct = c_df.groupby(c_df['date'].dt.date)['adv_pct'].agg(['mean', 'std', 'count']).to_dict('index')
    
    # Prepare data for line plot
    train_dates = []
    train_means = []
    train_errors = []
    
    test_dates = []
    test_means = []
    test_errors = []
    
    test2_dates = []
    test2_means = []
    test2_errors = []
    
    # Build the unified date set for ADV plot (daily granularity)
    all_adv_dates = set(train_daily_adv_pct.keys())
    if b_df is not None:
        all_adv_dates |= set(test_daily_adv_pct.keys())
    if c_df is not None:
        all_adv_dates |= set(test2_daily_adv_pct.keys())
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
        
        if c_df is not None and date in test2_daily_adv_pct:
            test2_dates.append(date)
            mean_val = test2_daily_adv_pct[date]['mean'] * 100  # Convert to percentage
            std_val = test2_daily_adv_pct[date]['std'] * 100 if test2_daily_adv_pct[date]['std'] is not None else 0
            count_val = test2_daily_adv_pct[date]['count']
            std_error = std_val / np.sqrt(count_val) if count_val > 0 else 0
            
            test2_means.append(mean_val)
            test2_errors.append(std_error)
    
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
    
    if c_df is not None and test2_dates:
        x_test2 = pd.to_datetime(test2_dates)
        axes[plot_idx].plot(x_test2, test2_means, '^-', color=COLOR['c'], label=c_name, linewidth=2, markersize=4)
        axes[plot_idx].fill_between(x_test2,
                                   [m - e for m, e in zip(test2_means, test2_errors)],
                                   [m + e for m, e in zip(test2_means, test2_errors)],
                                   color=COLOR['c'], alpha=0.2)
    
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
        weighted_cum_impact = 0
        rewards = []
        action_percentages = []
        slippages = []
        durations = []
        returns = []
        cum_impacts = []
        incomplete_orders = 0
        orders_with_no_arrival_price = 0
        total_orders = len(orders)
        
        # For trimming, we need to collect slippages with their notional weights
        slippage_notional_pairs = []
        
        for order_info in orders:
            # Get the first and last step info
            first_step = order_info[0]
            last_step = order_info[-1]
            
            # Compute executed notional (for weighting): filled_qty * order_vwap
            order_qty_val = first_step.get('order_qty', 0)
            shares_remaining_val = last_step.get('shares_remaining', 0)
            try:
                filled_qty = max(0, float(order_qty_val) - float(shares_remaining_val))
            except Exception:
                filled_qty = 0.0
            notional = float(last_step.get('order_vwap', 0.0)) * float(filled_qty)

            # Compute slippage (adverse sign) and intra-order return
            arrival_price = first_step.get('arrival_price', 0)
            side_val = first_step.get('side', None)
            if (side_val not in ('buy', 'sell')):
                logger.warning(
                    "Dropping order due to missing/invalid side: ticker=%s date=%s",
                    first_step.get('ticker', 'NA'), first_step.get('date', 'NA')
                )
                continue
            if arrival_price and arrival_price != 0:
                side_sign = 1.0 if side_val == 'buy' else -1.0
                slippage = side_sign * ((last_step['order_vwap'] - arrival_price) / arrival_price)
                raw_return = (last_step['mid_price'] - arrival_price) / arrival_price
                intra_return = raw_return if side_val == 'buy' else -raw_return
            else:
                slippage = 0.0
                intra_return = 0.0
                orders_with_no_arrival_price += 1

            # Accumulate all orders (filtering done upstream in _filter_orders_for_tca)
            total_notional += notional
            total_reward = last_step.get('total_reward', 0)
            rewards.append(total_reward)

            # Incomplete check (count only if kept)
            if last_step.get('shares_remaining', 0) > 0:
                incomplete_orders += 1

            weighted_return += intra_return * notional
            returns.append(intra_return)
            slippages.append(slippage)
            if notional > 0:
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
            
            # Collect cumulative impact (use last step); fallback to cost if needed
            cum_imp = last_step.get('accumulated_impact', None)
            if cum_imp is None:
                cum_imp = last_step.get('accumulated_impact_cost', None)
            try:
                if cum_imp is not None:
                    cum_imp_float = float(cum_imp)
                    weighted_cum_impact += cum_imp_float * notional
                    cum_impacts.append(cum_imp_float)
            except Exception:
                pass
        
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
            weighted_cum_impact /= total_notional
        
        # Calculate standard errors (using all data, not trimmed)
        n = len(slippages)
        slippage_stderr = (np.std(slippages) / np.sqrt(n)) * 10000 if n > 0 else 0  # Convert to bps
        duration_stderr = (np.std(durations) / np.sqrt(n)) if n > 0 else 0
        return_stderr = (np.std(returns) / np.sqrt(n)) * 10000 if n > 0 else 0  # Convert to bps
        action_stderr = (np.std(action_percentages) / np.sqrt(len(action_percentages))) * 100 if action_percentages else 0  # Convert to percentage
        cum_imp_stderr = (np.std(cum_impacts) / np.sqrt(len(cum_impacts))) * 10000 if cum_impacts else 0  # bps
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
            'Weighted Cumulative Impact (bps)': weighted_cum_impact * 10000,  # Convert to bps
            'Cum Impact Std Err (bps)': cum_imp_stderr,
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
        'Cum Impact (bps)',
        'Cum Impact SE (bps)',
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
        'Cum Impact (bps)': 'Weighted Cumulative Impact (bps)',
        'Cum Impact SE (bps)': 'Cum Impact Std Err (bps)',
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
    
    Note: This function expects pre-filtered data from _filter_orders_for_tca.
          If using create_execution_summary_and_plots, filtering is applied automatically.
          Otherwise, call _filter_orders_for_tca first.

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
    
    def weighted_mean_std(group_df):
        """Compute notional-weighted mean and std for a group."""
        valid = group_df['slippage'].notna() & group_df['notional'].notna() & (group_df['notional'] > 0)
        if not valid.any():
            return pd.Series({'mean': np.nan, 'std': np.nan, 'count': 0})
        
        df_valid = group_df[valid]
        slippages = df_valid['slippage'].values
        notionals = df_valid['notional'].values
        
        # Notional-weighted mean
        wmean = np.average(slippages, weights=notionals)
        
        # Notional-weighted std (using reliability weights formula)
        variance = np.average((slippages - wmean)**2, weights=notionals)
        wstd = np.sqrt(variance)
        
        return pd.Series({'mean': wmean, 'std': wstd, 'count': len(df_valid)})

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
            side = first_step.get('side', None)
            if side not in ('buy', 'sell'):
                # Skip rows with invalid side
                continue

            slippage = None
            if arrival_price and order_vwap:
                side_sign = 1.0 if side == 'buy' else -1.0
                slippage = side_sign * ((order_vwap - arrival_price) / arrival_price)

            intra_return = None
            if arrival_price and (last_step.get('mid_price', None) is not None):
                raw_return = (last_step['mid_price'] - arrival_price) / arrival_price
                # Side-adjust: positive means favorable move (up for buys, down for sells)
                intra_return = raw_return if side == 'buy' else -raw_return

            # Compute notional (order_qty * arrival_price)
            order_qty = first_step.get('order_qty', None)
            order_vwap = last_step.get('order_vwap', None)
            notional = None
            if order_qty and order_vwap:
                notional = float(order_qty) * float(order_vwap)
            
            # All filtering done upstream in _filter_orders_for_tca
            rows.append({
                'slippage': slippage,
                'notional': notional,
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

    # Build figure with 6 rows x 1 column for consistent layout
    from matplotlib.dates import DateFormatter
    fig, axes_all = plt.subplots(6, 1, figsize=(14, 50))  # Increased from 20 to 50 (2.5x)
    ax_top = axes_all[0]  # Date plot
    ax_box = axes_all[1]  # Model comparison
    axes = axes_all[2:6]  # 4 factor plots

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
            valid = df['slippage'].notna() & df['order_date'].notna() & df['notional'].notna() & (df['notional'] > 0)
            if not valid.any():
                continue
            dfv = df.loc[valid, ['order_date', 'slippage', 'notional']].copy()
            dfv['order_date'] = pd.to_datetime(dfv['order_date'], errors='coerce').dt.to_period('W').dt.start_time.dt.date
            dfv = dfv.dropna(subset=['order_date'])
            
            # Compute notional-weighted statistics by date
            stats = dfv.groupby('order_date').apply(weighted_mean_std)
            means = stats['mean'] * 10000.0
            stds = stats['std'] * 10000.0
            counts = stats['count']
            
            with np.errstate(divide='ignore', invalid='ignore'):
                se = (stds / np.sqrt(counts)).fillna(0.0)

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
            # Plot mean slippage with SE (no label here, will use shared legend)
            ax_top.errorbar(x_positions, means_full.values, yerr=se_full.values,
                            label=label_name, color=color_cycle[mi % len(color_cycle)],
                            marker='o', linestyle='-')

        # Format x-axis with dates
        ax_top.set_xticks(x_positions)
        ax_top.set_xticklabels([pd.to_datetime(d).strftime('%Y-%m-%d') for d in unique_weeks], rotation=45, ha='right')
        ax_top.grid(True, alpha=0.3)
    else:
        ax_top.text(0.5, 0.5, 'No date data to plot', ha='center', va='center', transform=ax_top.transAxes)

    # Model-wise line plot with standard error bands (replace boxplot) - NOTIONAL WEIGHTED
    ax_box.set_title("Arrival Slippage by Model (Notional-Weighted Mean ± SE, bps)")
    ax_box.set_ylabel("Arrival Slippage (bps)")
    ax_box.grid(True, alpha=0.3)
    x = np.arange(len(model_names), dtype=float)
    for i, model_name in enumerate(model_names):
        df = model_to_df[model_name]
        if df.empty or 'slippage' not in df.columns:
            continue
        
        # Compute notional-weighted mean and std
        valid = df['slippage'].notna() & df['notional'].notna() & (df['notional'] > 0)
        if not valid.any():
            continue
        
        df_valid = df[valid]
        slippages = df_valid['slippage'].values * 10000.0
        notionals = df_valid['notional'].values
        
        wmean = float(np.average(slippages, weights=notionals))
        variance = np.average((slippages - wmean)**2, weights=notionals)
        wstd = float(np.sqrt(variance))
        se = wstd / np.sqrt(len(df_valid)) if len(df_valid) > 1 else 0.0
        
        ax_box.plot([x[i]], [wmean], marker='o', color=color_cycle[i % len(color_cycle)], label=(model_name or "")[0:25])
        ax_box.fill_between([x[i]-0.25, x[i]+0.25], [wmean-se, wmean-se], [wmean+se, wmean+se], color=color_cycle[i % len(color_cycle)], alpha=0.2)
    ax_box.set_xticks(np.arange(len(model_names)))
    ax_box.set_xticklabels([(m or "")[0:25] for m in model_names], rotation=45, ha='right')  # Rotate labels 45 degrees

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
            valid = df['slippage'].notna() & df[factor].notna() & df['notional'].notna() & (df['notional'] > 0)
            dfv = df.loc[valid, [factor, 'slippage', 'notional']].copy()
            if dfv.empty:
                continue

            dfv['bin'] = pd.cut(dfv[factor], bins=bin_edges, include_lowest=True)
            
            # Compute notional-weighted statistics by bin
            stats = dfv.groupby('bin', observed=True).apply(weighted_mean_std)
            means = stats['mean'] * 10000.0
            stds = stats['std'] * 10000.0
            counts = stats['count']
            
            with np.errstate(divide='ignore', invalid='ignore'):
                se = (stds / np.sqrt(counts)).fillna(0.0)
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
        # No legend on individual plots

    # Hide unused axes if we have fewer than 4 factors
    for j in range(n_plots, len(axes)):
        axes[j].set_visible(False)

    # Create a single shared legend at the top of the figure
    # Collect handles and labels from the first plot (ax_top)
    handles, labels = ax_top.get_legend_handles_labels()
    if handles:
        fig.legend(handles, labels, 
                  loc='upper center', ncol=min(len(model_names), 6),
                  bbox_to_anchor=(0.5, 0.99), frameon=True, fontsize=10,
                  columnspacing=1.0, handletextpad=0.5)

    plt.tight_layout(rect=[0, 0, 1, 0.98])  # Leave space for legend at top
    plt.show()


def _filter_orders_for_tca(orders_dict, trim=0.01):
    """
    Filter and trim orders for TCA analysis (used by both table and plots).
    
    Filtering criteria:
      1. Drop orders with missing/zero arrival price
      2. Drop orders with invalid/missing daily_volatility (sigma)
      3. Drop orders with sigma >= 2.0 (200% daily volatility)
      4. Two-stage trimming:
         a. Trim top 1% by notional (remove largest trades)
         b. Trim top/bottom 1% by slippage (remove extreme slippage values)
    
    Note: After two-stage trimming, the table computes notional-weighted averages.
          Does NOT apply sigma-based slippage/return outlier filters.
    
    Args:
        orders_dict: Dict[model_name, list_of_orders]
        trim: Percentile to trim from each tail (0.01 = 1%)
    
    Returns:
        Filtered orders_dict with same structure
    """
    cleaned = {}
    for model_name, orders in orders_dict.items():
        # Robust emptiness check (avoid ambiguous truth values for pandas objects)
        is_empty = False
        try:
            is_empty = (orders is None) or (len(orders) == 0)
        except Exception:
            is_empty = True
        if is_empty:
            cleaned[model_name] = []
            continue
        # First pass: filter by sigma rules
        kept = []
        slippages_kept = []
        for order_info in orders:
            if order_info is None or len(order_info) == 0:
                continue
            first = order_info[0]
            last = order_info[-1]
            arrival = first.get('arrival_price', None)
            vwap = last.get('order_vwap', None)
            side = first.get('side', None)
            if side not in ('buy', 'sell'):
                continue
            if arrival in (None, 0, np.nan) or vwap in (None, np.nan):
                continue
            slippage = (vwap - arrival) / arrival
            if side == 'sell':
                slippage = -slippage

            dv = first.get('daily_volatility', None)
            dv_val = None
            if dv is not None:
                try:
                    dv_val = float(dv)
                except Exception:
                    dv_val = None
            if (dv_val is None) or (not np.isfinite(dv_val)) or (dv_val <= 0):
                dv1 = first.get('daily_volatility_lag1', None)
                if dv1 is not None:
                    try:
                        dv_val = float(dv1)
                    except Exception:
                        dv_val = None

            # compute intra-order return for logging
            last_mid = last.get('mid_price', None)
            intr = 0.0
            if (last_mid is not None) and (arrival not in (None, 0, np.nan)):
                raw = (last_mid - arrival) / arrival
                intr = raw if side == 'buy' else -raw

            # drop if bad sigma
            if (dv_val is None) or (not np.isfinite(dv_val)) or (dv_val <= 0) or (dv_val >= 2.0):
                logger.warning(
                    "Dropping order (sigma invalid or >=2.0): model=%s ticker=%s date=%s side=%s slippage_bps=%.1f sigma=%s",
                    model_name,
                    first.get('ticker', 'NA'),
                    first.get('date', 'NA'),
                    side,
                    f"{slippage * 10000.0:.1f}",
                    "NA" if dv_val is None else f"{dv_val:.4f}",
                )
                continue

            # Compute notional for weighting
            order_qty_val = first.get('order_qty', 0)
            shares_remaining_val = last.get('shares_remaining', 0)
            try:
                filled_qty = max(0, float(order_qty_val) - float(shares_remaining_val))
            except Exception:
                filled_qty = 0.0
            notional = float(vwap) * float(filled_qty)

            kept.append(order_info)
            if notional > 0:
                slippages_kept.append((slippage, notional, order_info))

        # Second pass: two-stage trimming (vectorized)
        if trim > 0 and len(slippages_kept) > 0:
            # Convert to numpy arrays for vectorized operations
            slippages = np.array([pair[0] for pair in slippages_kept])
            notionals = np.array([pair[1] for pair in slippages_kept])
            orders = [pair[2] for pair in slippages_kept]
            
            # Stage 1: Trim top 1% by notional (vectorized)
            notional_cutoff = np.quantile(notionals, 1 - trim)
            notional_mask = notionals <= notional_cutoff
            
            # Apply notional mask
            slippages_stage1 = slippages[notional_mask]
            notionals_stage1 = notionals[notional_mask]
            orders_stage1 = [orders[i] for i in np.where(notional_mask)[0]]
            
            # Stage 2: Trim top/bottom 1% by slippage (vectorized)
            if len(slippages_stage1) > 0:
                lower = np.quantile(slippages_stage1, trim)
                upper = np.quantile(slippages_stage1, 1 - trim)
                slippage_mask = (slippages_stage1 >= lower) & (slippages_stage1 <= upper)
                
                # Apply slippage mask
                trimmed = [orders_stage1[i] for i in np.where(slippage_mask)[0]]
                
                # Log statistics (optional)
                n_dropped_notional = (~notional_mask).sum()
                n_dropped_slippage = (~slippage_mask).sum()
                if n_dropped_notional > 0 or n_dropped_slippage > 0:
                    logger.debug(
                        "Trimming model=%s: dropped %d by notional (>%.0f), %d by slippage (outside [%.1f, %.1f] bps)",
                        model_name,
                        n_dropped_notional,
                        notional_cutoff,
                        n_dropped_slippage,
                        lower * 10000.0,
                        upper * 10000.0,
                    )
                
                cleaned[model_name] = trimmed
            else:
                cleaned[model_name] = []
        else:
            cleaned[model_name] = kept

    return cleaned


def create_execution_summary_and_plots(orders_dict, trim=0.01):
    """
    Print the execution summary table and render plots.
    
    Applies unified filtering once, then passes cleaned data to table and plots.
    """
    # Apply unified filtering: drop bad data + notional-weighted trimming
    filtered = _filter_orders_for_tca(orders_dict, trim=trim)
    
    # Pass filtered data to table and plots
    create_execution_summary_table(filtered, trim=0)  # No additional trimming in table
    plot_arrival_slippage_by_factors(filtered)
    plot_actions_vs_normalized_horizon_with_returns(filtered)
    plot_orders(filtered, num_orders=3)


def plot_actions_vs_normalized_horizon_with_returns(orders_dict, models=None, num_bins=10):
    """
    Plot actions (%) for multiple models across normalized horizon bins.

    - Normalized horizon: current_step / time_horizon
    - num_bins: number of equal-width bins on [0, 1]
    - Supports comparing multiple models (e.g., CNN, MLP, TCN)

    Args:
        orders_dict: Dict[str, list[list[dict]]]
        models: Optional[list[str]] list of model names to plot.
                If None, auto-selects CNN, MLP, TCN models
        num_bins: int number of bins (default 10)
    
    Example:
        # Auto-detect
        plot_actions_vs_normalized_horizon_with_returns(orders_dict)
        
        # Specify models
        plot_actions_vs_normalized_horizon_with_returns(
            orders_dict,
            models=['ppo_cnn_...', 'ppo_mlp_...', 'ppo_tcn_...']
        )
    """

    model_names = list(orders_dict.keys())
    
    # Auto-select CNN, MLP, TCN models if not specified
    if models is None:
        models_to_plot = []
        
        # Look for specific model types in order
        for model_type in ['cnn', 'mlp', 'tcn']:
            matching = [m for m in model_names if model_type in (m or '').lower()]
            if matching:
                models_to_plot.append(matching[0])
        
        # If we didn't find specific types, just use first few models
        if len(models_to_plot) == 0:
            models_to_plot = model_names[:3]
    else:
        # Use provided models, filtering out any that don't exist
        models_to_plot = [m for m in models if m in orders_dict]
    
    if len(models_to_plot) < 2:
        print(f"plot_actions_vs_normalized_horizon: need at least 2 models to compare")
        print(f"  Found: {models_to_plot}")
        print(f"  Available: {model_names}")
        return

    bins = np.linspace(0.0, 1.0, num_bins + 1)
    bin_centers = (bins[:-1] + bins[1:]) * 0.5

    def collect_actions_by_bin(orders):
        per_bin_actions = [[] for _ in range(num_bins)]
        per_bin_returns = [[] for _ in range(num_bins)]
        for order_info in orders:
            if order_info is None:
                continue
            try:
                if len(order_info) == 0:
                    continue
            except Exception:
                pass
            first = order_info[0]
            horizon = first.get('time_horizon', None)
            if horizon is None or horizon <= 0:
                continue
            for step in order_info:
                if 'current_step' not in step:
                    continue
                # Get action as percentage; fallback to other keys if needed
                if 'action_percentage' in step and step['action_percentage'] is not None:
                    action_pct = step['action_percentage'] * 100.0
                elif 'last_action_fraction' in step and step['last_action_fraction'] is not None:
                    action_pct = float(step['last_action_fraction']) * 100.0
                elif 'action' in step and step['action'] is not None:
                    # assume in [0,1]; if already [0,100], min with 100
                    val = float(step['action'])
                    action_pct = val * 100.0 if val <= 1.0 else min(val, 100.0)
                else:
                    continue
                norm = float(step['current_step']) / float(horizon)
                if not np.isfinite(norm):
                    continue
                # Clamp to [0, 1]
                norm = max(0.0, min(1.0, norm))
                # Right-open except last bin
                idx = int(np.digitize(norm, bins, right=False) - 1)
                if idx < 0:
                    idx = 0
                if idx >= num_bins:
                    idx = num_bins - 1
                per_bin_actions[idx].append(action_pct)
                # side-adjusted return per step relative to arrival (per-order first-step arrival/side)
                arrival = first.get('arrival_price', None)
                side = first.get('side', None)
                if side not in ('buy', 'sell'):
                    continue
                mid = step.get('mid_price', None)
                if arrival not in (None, 0, np.nan) and mid not in (None, np.nan):
                    raw = (float(mid) - float(arrival)) / float(arrival)
                    sret = raw if side == 'buy' else -raw
                    per_bin_returns[idx].append(sret * 10000.0)
        return per_bin_actions, per_bin_returns

    # Collect data for all models
    model_data = {}
    for model in models_to_plot:
        bins_data, returns_data = collect_actions_by_bin(orders_dict[model])
        model_data[model] = {'bins': bins_data, 'returns': returns_data}

    # Prepare colors
    color_cycle = plt.rcParams['axes.prop_cycle'].by_key().get('color', [
        'tab:blue', 'tab:orange', 'tab:green', 'tab:red', 'tab:purple',
        'tab:brown', 'tab:pink', 'tab:gray', 'tab:olive', 'tab:cyan'
    ])

    # Compute mean and standard error per bin for each model
    def compute_stats(per_bin_lists):
        means = []
        ses = []
        counts = []
        for lst in per_bin_lists:
            arr = np.array(lst, dtype=float)
            arr = arr[np.isfinite(arr)]
            n = arr.size
            counts.append(n)
            if n == 0:
                means.append(np.nan)
                ses.append(0.0)
            else:
                m = float(np.mean(arr))
                s = float(np.std(arr, ddof=1)) if n > 1 else 0.0
                se = s / np.sqrt(n) if n > 0 else 0.0
                means.append(m)
                ses.append(se)
        return np.array(means, dtype=float), np.array(ses, dtype=float), np.array(counts, dtype=float)

    # Compute stats for all models
    model_stats = {}
    for model in models_to_plot:
        ma, sea, ca = compute_stats(model_data[model]['bins'])
        mra, sra, _ = compute_stats(model_data[model]['returns'])
        model_stats[model] = {'mean': ma, 'se': sea, 'count': ca, 'ret_mean': mra, 'ret_se': sra}

    # Helper to compute side-adjusted intra-order return per order
    def side_adjusted_return(order_info):
        if order_info is None or len(order_info) == 0:
            return np.nan
        first = order_info[0]
        last = order_info[-1]
        arrival = first.get('arrival_price', None)
        last_mid = last.get('mid_price', None)
        if arrival in (None, 0, np.nan) or last_mid in (None, np.nan):
            return np.nan
        raw = (last_mid - arrival) / arrival
        side = first.get('side', None)
        if side not in ('buy', 'sell'):
            return np.nan
        return raw if side == 'buy' else -raw

    # Collect returns for tertile split across all models
    returns_all = []
    model_returns = {}
    for model in models_to_plot:
        model_returns[model] = []
        for oi in orders_dict[model]:
            r = side_adjusted_return(oi)
            model_returns[model].append(r)
            if np.isfinite(r):
                returns_all.append(r)

    # Compute tertile thresholds
    if len(returns_all) >= 3:
        lo = float(np.nanpercentile(returns_all, 33.333))
        hi = float(np.nanpercentile(returns_all, 66.667))
    else:
        lo, hi = -np.inf, np.inf

    # Partition orders by tertile
    def filter_orders_by_tertile(orders, returns, bucket):
        subset = []
        for oi, r in zip(orders, returns):
            if not np.isfinite(r):
                continue
            if bucket == 'negative' and r < lo:
                subset.append(oi)
            elif bucket == 'neutral' and (r >= lo and r <= hi):
                subset.append(oi)
            elif bucket == 'positive' and r > hi:
                subset.append(oi)
        return subset

    # For a given subset, recompute binned action stats
    def stats_for_subset(subset_a, subset_b):
        a_binned, a_ret = collect_actions_by_bin(subset_a)
        b_binned, b_ret = collect_actions_by_bin(subset_b)
        ma_s, sea_s, _ = compute_stats(a_binned)
        mb_s, seb_s, _ = compute_stats(b_binned)
        mra_s, sra_s, _ = compute_stats(a_ret)
        mrb_s, srb_s, _ = compute_stats(b_ret)
        return ma_s, sea_s, mb_s, seb_s, mra_s, sra_s, mrb_s, srb_s

    # Get short labels for models
    model_labels = {m: (m or "")[:25] for m in models_to_plot}

    # Build 4 rows × 1 column layout: overall + positive + neutral + negative
    fig, axes = plt.subplots(4, 1, figsize=(14, 16), sharex=True)
    
    # Helper function to compute stats for filtered subsets
    def stats_for_subset_multi(models_list, subset_dict):
        """Compute stats for all models in a given tertile subset"""
        stats_dict = {}
        for model in models_list:
            subset = subset_dict[model]
            binned, ret = collect_actions_by_bin(subset)
            ma_s, sea_s, _ = compute_stats(binned)
            mra_s, sra_s, _ = compute_stats(ret)
            stats_dict[model] = {'mean': ma_s, 'se': sea_s, 'ret_mean': mra_s, 'ret_se': sra_s}
        return stats_dict

    def plot_panel(ax, title, stats_dict):
        """Plot panel with multiple models, no individual legends"""
        ax.set_title(title, fontsize=12, pad=10)
        ax.set_ylabel("Action (%)", fontsize=11)
        ax.grid(True, alpha=0.3)
        ax.set_xlim(0.0, 1.0)
        ax.set_xticks(bin_centers)
        
        # Plot each model
        for idx, model in enumerate(models_to_plot):
            color = color_cycle[idx % len(color_cycle)]
            label = model_labels[model]
            
            ma_p = stats_dict[model]['mean']
            sea_p = stats_dict[model]['se']
            
            # Main axis: actions
            ax.plot(bin_centers, ma_p, color=color, marker='o', label=label, linewidth=2)
            ax.fill_between(bin_centers, ma_p - sea_p, ma_p + sea_p, color=color, alpha=0.2)
        
        # Secondary axis: returns
        ax2 = ax.twinx()
        ax2.set_ylabel("Side-Adj Return (bps)", fontsize=11)
        
        for idx, model in enumerate(models_to_plot):
            color = color_cycle[idx % len(color_cycle)]
            label = model_labels[model]
            
            ra_p = stats_dict[model]['ret_mean']
            sra_p = stats_dict[model]['ret_se']
            
            ax2.plot(bin_centers, ra_p, color=color, linestyle='--', alpha=0.6, label=f"{label} ret")
            ax2.fill_between(bin_centers, ra_p - sra_p, ra_p + sra_p, color=color, alpha=0.15)
        
        return ax2  # Return secondary axis for legend extraction

    # Plot overall panel
    ax0 = plot_panel(axes[0], "Actions by Normalized Horizon (Mean ± SE) — Overall", model_stats)
    
    # Create filtered subsets for each tertile
    positive_subsets = {m: filter_orders_by_tertile(orders_dict[m], model_returns[m], 'positive') 
                       for m in models_to_plot}
    neutral_subsets = {m: filter_orders_by_tertile(orders_dict[m], model_returns[m], 'neutral') 
                      for m in models_to_plot}
    negative_subsets = {m: filter_orders_by_tertile(orders_dict[m], model_returns[m], 'negative') 
                       for m in models_to_plot}
    
    # Compute stats for each tertile
    positive_stats = stats_for_subset_multi(models_to_plot, positive_subsets)
    neutral_stats = stats_for_subset_multi(models_to_plot, neutral_subsets)
    negative_stats = stats_for_subset_multi(models_to_plot, negative_subsets)
    
    # Plot tertile panels
    plot_panel(axes[1], "Positive Side-Adj Return (top tertile)", positive_stats)
    plot_panel(axes[2], "Neutral Side-Adj Return (middle tertile)", neutral_stats)
    plot_panel(axes[3], "Negative Side-Adj Return (bottom tertile)", negative_stats)
    
    axes[3].set_xlabel("Normalized Horizon", fontsize=11)
    
    # Create a single shared legend at the top of the figure
    # Collect handles and labels from the first panel
    lines1, labels1 = axes[0].get_legend_handles_labels()
    # Get secondary axis from first panel
    for child in axes[0].get_children():
        if hasattr(child, 'get_legend_handles_labels'):
            try:
                lines2, labels2 = child.get_legend_handles_labels()
                if lines2:  # Found the secondary axis
                    break
            except:
                pass
    else:
        lines2, labels2 = [], []
    
    # Place legend above all subplots
    fig.legend(lines1 + lines2, labels1 + labels2, 
              loc='upper center', ncol=min(len(models_to_plot), 6), 
              bbox_to_anchor=(0.5, 0.98), frameon=True, fontsize=10,
              columnspacing=1.0, handletextpad=0.5)
    
    plt.tight_layout(rect=[0, 0, 1, 0.97])  # Leave space for legend at top
    plt.show()