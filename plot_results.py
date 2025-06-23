import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# print limit
pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', 5)
pd.set_option('display.width', None)
pd.set_option('display.max_colwidth', None)

# Configuration: Enable/disable specific plots
PLOTS_TO_GENERATE = {
    'mean_execution_time': True,     # Plot 1: Mean execution time vs process count
    'speedup_analysis': True,        # Plot 2: Speedup calculation
    'scatter_combined': True,        # Plot 3: Combined scatter with mean and median
    'scatter_mean_only': True,      # Plot 3a: Scatter with mean analysis only
    'scatter_median_only': True,    # Plot 3b: Scatter with median analysis only
    'box_plot': True,                # Plot 4: Box plot distribution
    'configurable_speedup': True     # Plot 5: Speedup with configurable reference
}

# Speedup reference configuration
SPEEDUP_REFERENCE = {
    'use_process_count': 2,  # Set the reference: from specific process count, or 'min' for minimum, 'max' for maximum
    'use_metric': 'mean'     # 'mean' or 'median'
}

# Load the CSV file
csv_file = '/home/leonardo/hpc/hpc_project/mpi_timing_results.csv'
save_image_path1 = '/home/leonardo/hpc/hpc_project/mpi_performance_mean.png'
save_image_path2 = '/home/leonardo/hpc/hpc_project/mpi_speedup_analysis.png'
save_scatter_path = '/home/leonardo/hpc/hpc_project/task_distribution_scatter.png'
df = pd.read_csv(csv_file)

# Group by 'processes' and calculate mean execution time
df_grouped = df.groupby('processes')['execution_time'].mean().reset_index()

# ------------------------------------------------------
# Plot 1: Mean execution time vs process count (separate figure)
# ------------------------------------------------------

if PLOTS_TO_GENERATE['mean_execution_time']:
    fig1, ax1 = plt.subplots(1, 1, figsize=(10, 6))
    ax1.plot(df_grouped['processes'], df_grouped['execution_time'], 'bo-', linewidth=2, markersize=8)

    # Add grid size annotations
    for i, row in df_grouped.iterrows():
        proc = row['processes']
        time = row['execution_time']
        grid_info = df[df['processes'] == proc].iloc[0]
        grid_label = f"{grid_info['grid_rows']}x{grid_info['grid_cols']}"
        ax1.annotate(grid_label, (proc, time), textcoords="offset points", 
                    xytext=(0,10), ha='center', fontsize=9)

    ax1.set_xlabel('Number of Processes')
    ax1.set_ylabel('Mean Execution Time (seconds)')
    ax1.set_title('Mean Execution Time vs Process Count (with Grid Size)')
    ax1.grid(True, alpha=0.3)

    # Force x-axis to show all process counts
    all_process_counts = sorted(df['processes'].unique())
    ax1.set_xticks(all_process_counts)
    ax1.set_xlim(min(all_process_counts) - 0.5, max(all_process_counts) + 0.5)

    plt.tight_layout()
    plt.savefig(save_image_path1, dpi=300, bbox_inches='tight')
    plt.show()

# ------------------------------------------------------
# Plot 2: Speedup calculation (separate figure)
# ------------------------------------------------------

if PLOTS_TO_GENERATE['speedup_analysis']:
    fig2, ax2 = plt.subplots(1, 1, figsize=(10, 6))
    if len(df_grouped) > 1:
        serial_time = df_grouped.iloc[0]['execution_time']
        speedup = serial_time / df_grouped['execution_time']
        ax2.plot(df_grouped['processes'], speedup, 'ro-', linewidth=2, markersize=8, label='Actual')
        ax2.plot(df_grouped['processes'], df_grouped['processes'], 'g--', linewidth=2, label='Ideal')
        ax2.set_xlabel('Number of Processes')
        ax2.set_ylabel('Speedup')
        ax2.set_title('Speedup vs Process Count')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        # Force x-axis to show all process counts
        all_process_counts = sorted(df['processes'].unique())
        ax2.set_xticks(all_process_counts)
        ax2.set_xlim(min(all_process_counts) - 0.5, max(all_process_counts) + 0.5)
    else:
        ax2.text(0.5, 0.5, 'Need multiple process\ncounts for speedup', 
                 ha='center', va='center', transform=ax2.transAxes)
        ax2.set_title('Speedup Analysis (Insufficient Data)')

    plt.tight_layout()
    plt.savefig(save_image_path2, dpi=300, bbox_inches='tight')
    plt.show()

# ------------------------------------------------------
# Plot 3: Scatter plot (already separate)
# ------------------------------------------------------

if PLOTS_TO_GENERATE['scatter_combined']:
    scatter_plotsize = (20, 8)

    fig3, ax3 = plt.subplots(1, 1, figsize=scatter_plotsize)
    fig3.suptitle('Task Distribution Analysis', fontsize=16)

    # Create scatter plot for each individual record
    for proc_count in sorted(df['processes'].unique()):
        subset = df[df['processes'] == proc_count]
        ax3.scatter([proc_count] * len(subset), subset['execution_time'], 
                   alpha=0.6, s=60, label=f'{proc_count} processes')

    # Add aggregated statistics at the bottom
    stats_by_process = df.groupby('processes')['execution_time'].agg(['count', 'mean', 'median']).reset_index()

    ####### regression analysis #######
    # Add 5th order polynomial regression curve for mean
    x_reg = stats_by_process['processes'].values
    y_reg_mean = stats_by_process['mean'].values
    y_reg_median = stats_by_process['median'].values

    # Fit 5th order polynomial for mean (ensure we have enough points)
    if len(x_reg) >= 6:  # Need at least 6 points for 5th order
        poly_coeffs_mean = np.polyfit(x_reg, y_reg_mean, 5)
        poly_coeffs_median = np.polyfit(x_reg, y_reg_median, 5)
    else:
        # Use lower order if not enough points
        order = min(5, len(x_reg) - 1) if len(x_reg) > 1 else 1
        poly_coeffs_mean = np.polyfit(x_reg, y_reg_mean, order)
        poly_coeffs_median = np.polyfit(x_reg, y_reg_median, order)

    # Generate smooth curve for plotting
    x_smooth = np.linspace(x_reg.min(), x_reg.max(), 100)
    y_smooth_mean = np.polyval(poly_coeffs_mean, x_smooth)
    y_smooth_median = np.polyval(poly_coeffs_median, x_smooth)

    # Plot the regression curves
    order_used = min(5, len(x_reg)-1) if len(x_reg) > 1 else 1
    ax3.plot(x_smooth, y_smooth_mean, 'r-', linewidth=3, alpha=0.8, 
             label=f'{order_used}th Order Polynomial Fit (Mean Values)')
    ax3.plot(x_smooth, y_smooth_median, 'b-', linewidth=3, alpha=0.8, 
             label=f'{order_used}th Order Polynomial Fit (Median Values)')

    # Plot mean and median points more prominently
    ax3.scatter(x_reg, y_reg_mean, color='red', s=100, marker='D', 
               edgecolor='darkred', linewidth=2, label='Mean Values', zorder=5)
    ax3.scatter(x_reg, y_reg_median, color='blue', s=100, marker='s', 
               edgecolor='darkblue', linewidth=2, label='Median Values', zorder=5)
    ########

    y_min = ax3.get_ylim()[0]
    y_range = ax3.get_ylim()[1] - ax3.get_ylim()[0]
    text_y = y_min + 0.03 * y_range  # Changed from 0.05 to 0.02 to move up

    for _, row in stats_by_process.iterrows():
        proc = row['processes']
        mean_time = row['mean']
        median_time = row['median']
        count = row['count']
        ax3.text(proc, text_y, f'Count: {count}\nMean: {mean_time:.4f}s\nMed: {median_time:.4f}s', 
                 ha='center', va='top', fontsize=9, 
                 bbox=dict(boxstyle='round,pad=0.3', facecolor='lightblue', alpha=0.8))

    ax3.set_xlabel('Number of Tasks/Processes')
    ax3.set_ylabel('Execution Time (seconds)')
    ax3.set_title('Individual Records with Aggregated Statistics')
    ax3.grid(True, alpha=0.3)

    # Force x-axis to show all process counts
    all_process_counts = sorted(df['processes'].unique())
    ax3.set_xticks(all_process_counts)
    ax3.set_xlim(min(all_process_counts) - 0.5, max(all_process_counts) + 0.5)

    # Create custom legend with single entry for processes
    from matplotlib.lines import Line2D
    custom_legend_elements = [
        Line2D([0], [0], marker='o', color='w', markerfacecolor='blue', markersize=8, alpha=0.6, label='Single Task Records'),
        Line2D([0], [0], color='red', linewidth=3, alpha=0.8, label=f'{order_used}th Order Polynomial Fit (Mean Values)'),
        Line2D([0], [0], color='blue', linewidth=3, alpha=0.8, label=f'{order_used}th Order Polynomial Fit (Median Values)'),
        Line2D([0], [0], marker='D', color='w', markerfacecolor='red', markeredgecolor='darkred', 
               markersize=8, markeredgewidth=2, label='Mean Values'),
        Line2D([0], [0], marker='s', color='w', markerfacecolor='blue', markeredgecolor='darkblue', 
               markersize=8, markeredgewidth=2, label='Median Values')
    ]
    legend = ax3.legend(handles=custom_legend_elements, loc='upper right')

    # Adjust y-axis to accommodate text at bottom
    ax3.set_ylim(bottom=text_y - 0.05 * y_range)

    plt.tight_layout()
    plt.savefig(save_scatter_path, dpi=300, bbox_inches='tight')
    plt.show()

# ------------------------------------------------------
# Plot 3a: Scatter plot with Mean analysis
# ------------------------------------------------------

if PLOTS_TO_GENERATE['scatter_mean_only']:
    fig3a, ax3a = plt.subplots(1, 1, figsize=scatter_plotsize)
    fig3a.suptitle('Task Distribution Analysis - Mean Values', fontsize=16)

    # Create scatter plot for each individual record
    for proc_count in sorted(df['processes'].unique()):
        subset = df[df['processes'] == proc_count]
        ax3a.scatter([proc_count] * len(subset), subset['execution_time'], 
                   alpha=0.6, s=60, label=f'{proc_count} processes')

    # Add aggregated statistics at the bottom
    stats_by_process = df.groupby('processes')['execution_time'].agg(['count', 'mean', 'median']).reset_index()

    ####### regression analysis for mean #######
    x_reg = stats_by_process['processes'].values
    y_reg_mean = stats_by_process['mean'].values

    # Fit polynomial for mean
    if len(x_reg) >= 6:
        poly_coeffs_mean = np.polyfit(x_reg, y_reg_mean, 5)
        order_used = 5
    else:
        order_used = min(5, len(x_reg) - 1) if len(x_reg) > 1 else 1
        poly_coeffs_mean = np.polyfit(x_reg, y_reg_mean, order_used)

    # Generate smooth curve for plotting
    x_smooth = np.linspace(x_reg.min(), x_reg.max(), 100)
    y_smooth_mean = np.polyval(poly_coeffs_mean, x_smooth)

    # Plot the regression curve for mean
    ax3a.plot(x_smooth, y_smooth_mean, 'r-', linewidth=3, alpha=0.8, 
             label=f'{order_used}th Order Polynomial Fit (Mean Values)')

    # Plot mean points more prominently
    ax3a.scatter(x_reg, y_reg_mean, color='red', s=100, marker='D', 
               edgecolor='darkred', linewidth=2, label='Mean Values', zorder=5)

    y_min = ax3a.get_ylim()[0]
    y_range = ax3a.get_ylim()[1] - ax3a.get_ylim()[0]
    text_y = y_min - 0.1 * y_range

    for _, row in stats_by_process.iterrows():
        proc = row['processes']
        mean_time = row['mean']
        count = row['count']
        ax3a.text(proc, text_y, f'Count: {count}\nMean: {mean_time:.4f}s', 
                 ha='center', va='top', fontsize=9, 
                 bbox=dict(boxstyle='round,pad=0.3', facecolor='lightcoral', alpha=0.8))

    ax3a.set_xlabel('Number of Tasks/Processes')
    ax3a.set_ylabel('Execution Time (seconds)')
    ax3a.set_title('Individual Records with Mean Statistics')
    ax3a.grid(True, alpha=0.3)

    # Force x-axis to show all process counts
    all_process_counts = sorted(df['processes'].unique())
    ax3a.set_xticks(all_process_counts)
    ax3a.set_xlim(min(all_process_counts) - 0.5, max(all_process_counts) + 0.5)

    # Create custom legend for mean plot
    from matplotlib.lines import Line2D
    custom_legend_elements_mean = [
        Line2D([0], [0], marker='o', color='w', markerfacecolor='blue', markersize=8, alpha=0.6, label='Single Task Records'),
        Line2D([0], [0], color='red', linewidth=3, alpha=0.8, label=f'{order_used}th Order Polynomial Fit (Mean Values)'),
        Line2D([0], [0], marker='D', color='w', markerfacecolor='red', markeredgecolor='darkred', 
               markersize=8, markeredgewidth=2, label='Mean Values')
    ]
    legend = ax3a.legend(handles=custom_legend_elements_mean, loc='upper right')

    # Adjust y-axis to accommodate text at bottom
    ax3a.set_ylim(bottom=text_y - 0.05 * y_range)

    plt.tight_layout()
    save_scatter_mean_path = '/home/leonardo/hpc/hpc_project/task_distribution_scatter_mean.png'
    plt.savefig(save_scatter_mean_path, dpi=300, bbox_inches='tight')
    plt.show()

# ------------------------------------------------------
# Plot 3b: Scatter plot with Median analysis
# ------------------------------------------------------

if PLOTS_TO_GENERATE['scatter_median_only']:
    fig3b, ax3b = plt.subplots(1, 1, figsize=scatter_plotsize)
    fig3b.suptitle('Task Distribution Analysis - Median Values', fontsize=16)

    # Create scatter plot for each individual record
    for proc_count in sorted(df['processes'].unique()):
        subset = df[df['processes'] == proc_count]
        ax3b.scatter([proc_count] * len(subset), subset['execution_time'], 
                   alpha=0.6, s=60, label=f'{proc_count} processes')

    ####### regression analysis for median #######
    y_reg_median = stats_by_process['median'].values

    # Fit polynomial for median
    if len(x_reg) >= 6:
        poly_coeffs_median = np.polyfit(x_reg, y_reg_median, 5)
        order_used = 5
    else:
        order_used = min(5, len(x_reg) - 1) if len(x_reg) > 1 else 1
        poly_coeffs_median = np.polyfit(x_reg, y_reg_median, order_used)

    # Generate smooth curve for plotting
    y_smooth_median = np.polyval(poly_coeffs_median, x_smooth)

    # Plot the regression curve for median
    ax3b.plot(x_smooth, y_smooth_median, 'b-', linewidth=3, alpha=0.8, 
             label=f'{order_used}th Order Polynomial Fit (Median Values)')

    # Plot median points more prominently
    ax3b.scatter(x_reg, y_reg_median, color='blue', s=100, marker='s', 
               edgecolor='darkblue', linewidth=2, label='Median Values', zorder=5)

    y_min = ax3b.get_ylim()[0]
    y_range = ax3b.get_ylim()[1] - ax3b.get_ylim()[0]
    text_y = y_min - 0.1 * y_range

    for _, row in stats_by_process.iterrows():
        proc = row['processes']
        median_time = row['median']
        count = row['count']
        ax3b.text(proc, text_y, f'Count: {count}\nMed: {median_time:.4f}s', 
                 ha='center', va='top', fontsize=9, 
                 bbox=dict(boxstyle='round,pad=0.3', facecolor='lightblue', alpha=0.8))

    ax3b.set_xlabel('Number of Tasks/Processes')
    ax3b.set_ylabel('Execution Time (seconds)')
    ax3b.set_title('Individual Records with Median Statistics')
    ax3b.grid(True, alpha=0.3)

    # Force x-axis to show all process counts
    ax3b.set_xticks(all_process_counts)
    ax3b.set_xlim(min(all_process_counts) - 0.5, max(all_process_counts) + 0.5)

    # Create custom legend for median plot
    custom_legend_elements_median = [
        Line2D([0], [0], marker='o', color='w', markerfacecolor='blue', markersize=8, alpha=0.6, label='Single Task Records'),
        Line2D([0], [0], color='blue', linewidth=3, alpha=0.8, label=f'{order_used}th Order Polynomial Fit (Median Values)'),
        Line2D([0], [0], marker='s', color='w', markerfacecolor='blue', markeredgecolor='darkblue', 
               markersize=8, markeredgewidth=2, label='Median Values')
    ]
    legend = ax3b.legend(handles=custom_legend_elements_median, loc='upper right')

    # Adjust y-axis to accommodate text at bottom
    ax3b.set_ylim(bottom=text_y - 0.05 * y_range)

    plt.tight_layout()
    save_scatter_median_path = '/home/leonardo/hpc/hpc_project/task_distribution_scatter_median.png'
    plt.savefig(save_scatter_median_path, dpi=300, bbox_inches='tight')
    plt.show()

# ------------------------------------------------------
# Plot 4: Box plot for distribution analysis
# ------------------------------------------------------

if PLOTS_TO_GENERATE['box_plot']:
    fig4, ax4 = plt.subplots(1, 1, figsize=(12, 8))

    # Prepare data for box plot
    box_data = []
    box_labels = []
    process_counts = sorted(df['processes'].unique())

    for proc_count in process_counts:
        subset = df[df['processes'] == proc_count]['execution_time']
        box_data.append(subset.values)
        box_labels.append(str(proc_count))

    # Create box plot
    bp = ax4.boxplot(box_data, labels=box_labels, patch_artist=True, 
                     notch=True, showmeans=True, meanline=True)

    # Customize box plot colors
    colors = plt.cm.viridis(np.linspace(0, 1, len(box_data)))
    for patch, color in zip(bp['boxes'], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)

    # Customize other elements
    for element in ['whiskers', 'fliers', 'medians', 'caps']:
        plt.setp(bp[element], color='black')
        
    plt.setp(bp['means'], color='red', linewidth=2)

    ax4.set_xlabel('Number of Processes')
    ax4.set_ylabel('Execution Time (seconds)')
    ax4.set_title('Execution Time Distribution by Process Count (Box Plot)')
    ax4.grid(True, alpha=0.3)

    # Add legend
    from matplotlib.lines import Line2D
    legend_elements = [
        Line2D([0], [0], color='black', linewidth=1, label='Median'),
        Line2D([0], [0], color='red', linewidth=2, label='Mean'),
        Line2D([0], [0], marker='o', color='w', markerfacecolor='black', 
               markersize=4, label='Outliers', linestyle='None')
    ]
    ax4.legend(handles=legend_elements, loc='upper right')

    plt.tight_layout()
    save_boxplot_path = '/home/leonardo/hpc/hpc_project/mpi_distribution_boxplot.png'
    plt.savefig(save_boxplot_path, dpi=300, bbox_inches='tight')
    plt.show()

# ------------------------------------------------------
# Plot 5: Configurable speedup analysis
# ------------------------------------------------------

if PLOTS_TO_GENERATE['configurable_speedup']:
    fig5, ax5 = plt.subplots(1, 1, figsize=(12, 8))

    ax5.set_ylim(0, 5)  # Set y-axis limit for speedup factor
    
    # Calculate statistics for all process counts
    stats_by_process = df.groupby('processes')['execution_time'].agg(['mean', 'median']).reset_index()
    
    # Determine reference time based on configuration
    reference_proc = SPEEDUP_REFERENCE['use_process_count']
    reference_metric = SPEEDUP_REFERENCE['use_metric']
    
    if reference_proc == 'min':
        reference_proc_actual = stats_by_process['processes'].min()
    elif reference_proc == 'max':
        reference_proc_actual = stats_by_process['processes'].max()
    else:
        reference_proc_actual = reference_proc
    
    # Get reference time
    reference_row = stats_by_process[stats_by_process['processes'] == reference_proc_actual]
    if len(reference_row) == 0:
        print(f"Warning: Reference process count {reference_proc_actual} not found in data. Using minimum.")
        reference_proc_actual = stats_by_process['processes'].min()
        reference_row = stats_by_process[stats_by_process['processes'] == reference_proc_actual]
    
    reference_time = reference_row[reference_metric].iloc[0]
    
    # Calculate speedup for the selected metric only
    if reference_metric == 'mean':
        speedup_values = reference_time / stats_by_process['mean']
        plot_color = 'ro-'
        label_text = f'Speedup (Mean) - Ref: {reference_proc_actual} processes'
    else:  # median
        speedup_values = reference_time / stats_by_process['median']
        plot_color = 'bo-'
        label_text = f'Speedup (Median) - Ref: {reference_proc_actual} processes'
    
    # Plot speedup curve for selected metric only
    ax5.plot(stats_by_process['processes'], speedup_values, plot_color, linewidth=2, 
             markersize=8, label=label_text)
    
    ax5.set_xlabel('Number of Processes/Tasks')
    ax5.set_ylabel('Speedup Factor')
    ax5.set_title(f'Speedup Analysis (Reference: {reference_proc_actual} processes, {reference_metric} time)')
    ax5.grid(True, alpha=0.3)
    ax5.legend()
    
    # Force x-axis to show all process counts
    all_process_counts = sorted(df['processes'].unique())
    ax5.set_xticks(all_process_counts)
    ax5.set_xlim(min(all_process_counts) - 0.5, max(all_process_counts) + 0.5)
    
    # Add reference line at speedup = 1
    ax5.axhline(y=1, color='black', linestyle=':', alpha=0.5, label='No speedup')
    
    plt.tight_layout()
    save_speedup_config_path = '/home/leonardo/hpc/hpc_project/mpi_configurable_speedup.png'
    plt.savefig(save_speedup_config_path, dpi=300, bbox_inches='tight')
    plt.show()
    
    # Print speedup summary for selected metric only
    print(f"\n" + "="*60)
    print(f"CONFIGURABLE SPEEDUP ANALYSIS")
    print(f"Reference: {reference_proc_actual} processes, {reference_metric} = {reference_time:.6f}s")
    print("="*60)
    for i, row in stats_by_process.iterrows():
        proc = int(row['processes'])  # Convert to int
        speedup = speedup_values.iloc[i]
        print(f"Processes: {proc:2d} | Speedup({reference_metric.title()}): {speedup:.3f}x")

# ------------------------------------------------------------------
# Report performance metrics with print statements
# ------------------------------------------------------------------

# Display basic information about the data
print("Data overview:")
print(df)
print(f"\nData shape: {df.shape}")
print(f"Unique process counts: {sorted(df['processes'].unique())}")
print(f"\nGrid size range: {df['grid_rows'].min()}x{df['grid_cols'].min()} to {df['grid_rows'].max()}x{df['grid_cols'].max()}")

# Display performance metrics
print("\n" + "="*50)
print("PERFORMANCE METRICS")
print("="*50)

# Calculate speedup values
baseline_time = None
for proc_count in sorted(df['processes'].unique()):
    subset = df[df['processes'] == proc_count]
    avg_time = subset['execution_time'].mean()
    std_time = subset['execution_time'].std()
    count = len(subset)
    
    # Set baseline (first/smallest process count) for speedup calculation
    if baseline_time is None:
        baseline_time = avg_time
        speedup = 1.0
    else:
        speedup = baseline_time / avg_time
    
    print(f"Processes: {proc_count:2d} | Count: {count:2d} | Avg Time: {avg_time:.6f}s | Std Dev: {std_time:.6f}s | Speedup: {speedup:.3f}x")

# Additional speedup analysis
print("\n" + "="*25)
print("SPEEDUP ANALYSIS")
print("="*25)
print(f"Baseline time (processes={sorted(df['processes'].unique())[0]}): {baseline_time:.6f}s")
for proc_count in sorted(df['processes'].unique()):
    subset = df[df['processes'] == proc_count]
    avg_time = subset['execution_time'].mean()
    speedup = baseline_time / avg_time
    efficiency = speedup / proc_count * 100
    print(f"Processes: {proc_count:2d} | Speedup: {speedup:.3f}x | Efficiency: {efficiency:.1f}%")
