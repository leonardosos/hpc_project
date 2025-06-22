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
save_image_path = '/home/leonardo/hpc/hpc_project/mpi_performance_analysis.png'
save_scatter_path = '/home/leonardo/hpc/hpc_project/task_distribution_scatter.png'
df = pd.read_csv(csv_file)

# Display basic information about the data
print("Data overview:")
print(df)
print(f"\nData shape: {df.shape}")
print(f"Unique process counts: {sorted(df['processes'].unique())}")
print(f"\nGrid size range: {df['grid_rows'].min()}x{df['grid_cols'].min()} to {df['grid_rows'].max()}x{df['grid_cols'].max()}")

# Create main visualizations
fig, axes = plt.subplots(1, 2, figsize=(15, 5))
fig.suptitle('MPI Performance Analysis', fontsize=16)

# Plot 1: Execution time vs number of processes with grid size values
ax1 = axes[0]
df_grouped = df.groupby('processes')['execution_time'].mean().reset_index()
ax1.plot(df_grouped['processes'], df_grouped['execution_time'], 'bo-', linewidth=2, markersize=8)

# Add grid size annotations
for i, row in df_grouped.iterrows():
    proc = row['processes']
    time = row['execution_time']
    # Get corresponding grid size for this process count
    grid_info = df[df['processes'] == proc].iloc[0]
    grid_label = f"{grid_info['grid_rows']}x{grid_info['grid_cols']}"
    ax1.annotate(grid_label, (proc, time), textcoords="offset points", 
                xytext=(0,10), ha='center', fontsize=9)

ax1.set_xlabel('Number of Processes')
ax1.set_ylabel('Mean Execution Time (seconds)')
ax1.set_title('Mean Execution Time vs Process Count (with Grid Size)')
ax1.grid(True, alpha=0.3)

# Plot 2: Speedup calculation (if we have multiple process counts)
ax2 = axes[1]
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
else:
    ax2.text(0.5, 0.5, 'Need multiple process\ncounts for speedup', 
             ha='center', va='center', transform=ax2.transAxes)
    ax2.set_title('Speedup Analysis (Insufficient Data)')

plt.tight_layout()
plt.savefig(save_image_path, dpi=300, bbox_inches='tight')
plt.show()

# Create dedicated scatter plot visualization
fig2, ax3 = plt.subplots(1, 1, figsize=(10, 8))
fig2.suptitle('Task Distribution Analysis', fontsize=16)

# Create scatter plot for each individual record
for proc_count in sorted(df['processes'].unique()):
    subset = df[df['processes'] == proc_count]
    ax3.scatter([proc_count] * len(subset), subset['execution_time'], 
               alpha=0.6, s=60, label=f'{proc_count} processes')

# Add aggregated statistics at the bottom
stats_by_process = df.groupby('processes')['execution_time'].agg(['count', 'mean']).reset_index()
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
ax3.legend()

# Adjust y-axis to accommodate text at bottom
ax3.set_ylim(bottom=text_y - 0.05 * y_range)

plt.tight_layout()
plt.savefig(save_scatter_path, dpi=300, bbox_inches='tight')
plt.show()

# Print performance metrics
print("\n" + "="*50)
print("PERFORMANCE METRICS")
print("="*50)
for proc_count in sorted(df['processes'].unique()):
    subset = df[df['processes'] == proc_count]
    avg_time = subset['execution_time'].mean()
    std_time = subset['execution_time'].std()
    count = len(subset)
    print(f"Processes: {proc_count:2d} | Count: {count:2d} | Avg Time: {avg_time:.6f}s | Std Dev: {std_time:.6f}s")
