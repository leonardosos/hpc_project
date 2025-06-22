import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# print limit
pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', 5)
pd.set_option('display.width', None)
pd.set_option('display.max_colwidth', None)

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

fig3, ax3 = plt.subplots(1, 1, figsize=(20, 8))
fig3.suptitle('Task Distribution Analysis', fontsize=16)

# Create scatter plot for each individual record
for proc_count in sorted(df['processes'].unique()):
    subset = df[df['processes'] == proc_count]
    ax3.scatter([proc_count] * len(subset), subset['execution_time'], 
               alpha=0.6, s=60, label=f'{proc_count} processes')

# Add aggregated statistics at the bottom
stats_by_process = df.groupby('processes')['execution_time'].agg(['count', 'mean']).reset_index()

####### regression analysis #######
# Add 5th order polynomial regression curve
x_reg = stats_by_process['processes'].values
y_reg = stats_by_process['mean'].values

# Fit 5th order polynomial (ensure we have enough points)
if len(x_reg) >= 6:  # Need at least 6 points for 5th order
    poly_coeffs = np.polyfit(x_reg, y_reg, 5)
else:
    # Use lower order if not enough points
    order = min(5, len(x_reg) - 1) if len(x_reg) > 1 else 1
    poly_coeffs = np.polyfit(x_reg, y_reg, order)

# Generate smooth curve for plotting
x_smooth = np.linspace(x_reg.min(), x_reg.max(), 100)
y_smooth = np.polyval(poly_coeffs, x_smooth)

# Plot the regression curve
ax3.plot(x_smooth, y_smooth, 'r-', linewidth=3, alpha=0.8, 
         label=f'{min(5, len(x_reg)-1) if len(x_reg) > 1 else 1}th Order Polynomial Fit (Mean Values)')

# Plot mean points more prominently
ax3.scatter(x_reg, y_reg, color='red', s=100, marker='D', 
           edgecolor='darkred', linewidth=2, label='Mean Values', zorder=5)
########

y_min = ax3.get_ylim()[0]
y_range = ax3.get_ylim()[1] - ax3.get_ylim()[0]
text_y = y_min - 0.1 * y_range

for _, row in stats_by_process.iterrows():
    proc = row['processes']
    mean_time = row['mean']
    count = row['count']
    ax3.text(proc, text_y, f'Count: {count}\nMean: {mean_time:.4f}s', 
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
    Line2D([0], [0], color='red', linewidth=3, alpha=0.8, label=f'{min(5, len(x_reg)-1) if len(x_reg) > 1 else 1}th Order Polynomial Fit (Mean Values)'),
    Line2D([0], [0], marker='D', color='w', markerfacecolor='red', markeredgecolor='darkred', 
           markersize=8, markeredgewidth=2, label='Mean Values')
]
legend = ax3.legend(handles=custom_legend_elements, loc='upper right')


# Adjust y-axis to accommodate text at bottom
ax3.set_ylim(bottom=text_y - 0.05 * y_range)

plt.tight_layout()
plt.savefig(save_scatter_path, dpi=300, bbox_inches='tight')
plt.show()

"""
# ------------------------------------------------------
# Plot 4: Box plot for distribution analysis
# ------------------------------------------------------

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
"""
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
for proc_count in sorted(df['processes'].unique()):
    subset = df[df['processes'] == proc_count]
    avg_time = subset['execution_time'].mean()
    std_time = subset['execution_time'].std()
    count = len(subset)
    print(f"Processes: {proc_count:2d} | Count: {count:2d} | Avg Time: {avg_time:.6f}s | Std Dev: {std_time:.6f}s")
