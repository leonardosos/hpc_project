# basic library imports
from mpi4py import MPI
import numpy as np
import time

# libraries for report file 
import csv
from datetime import datetime
import os

# Start timing the entire script
start_time = time.time()

# ------------------------------------------------------------------
# 1) Initialize MPI and create a 2D Cartesian communicator
# ------------------------------------------------------------------

# Initialize MPI
# Get the global communicator and its rank and size
world_comm = MPI.COMM_WORLD
world_rank = world_comm.Get_rank()
world_size = world_comm.Get_size()

# Define the dimensions of the Cartesian grid
dims = MPI.Compute_dims(world_size, 2)
periods = (True, True)    # both dimensions are periodic
reorder = True            # allow reordering of ranks

# Create the Cartesian communicator
cart_comm = world_comm.Create_cart(dims, periods=periods, reorder=reorder)

# Get the rank and coordinates in the Cartesian communicator
cart_rank = cart_comm.Get_rank()
coords    = cart_comm.Get_coords(cart_rank)


# Step 1: compare local (cart) rank vs. global (world) rank
same_rank = (cart_rank == world_rank)
same = "same" if same_rank else "different"
print(f"[World rank {world_rank:2d}] "
      f"[Cart rank {cart_rank:2d}]"
      f" :: {same} rank :: "
      f"--> coords {coords} "
      f"of {dims[0]}x{dims[1]} grid", flush=True)


# ------------------------------------------------------------------
# 2) Exchange ranks with N, S, E, W neighbours and compute the 5‐point average
# ------------------------------------------------------------------

# Pre-allocated arrays 
nbr_buf = np.empty(4, dtype=np.int32) # Buffer for neighbours
my_buf  = np.array([cart_rank], dtype=np.int32) # Buffer for self rank

# Define the directions and their corresponding shifts
# Directions: (axis, displacement, index in nbr_buf)
directions = {
    'north': (0, -1, 0),
    'south': (0, +1, 1), 
    'east':  (1, +1, 2),
    'west':  (1, -1, 3)
}

# Sendrecv to exchange ranks with neighbours
for direction, (axis, disp, idx) in directions.items():
    # Get source and destination ranks for the shift
    src, dst = cart_comm.Shift(axis, disp)

    cart_comm.Sendrecv(
        sendbuf=my_buf[0:1],            # send local rank (cart_rank)
        dest=dst,                       # send to next rank in the specified direction
        recvbuf=nbr_buf[idx:idx+1],     # receive neighbour rank and store in nbr_buf
        source=src,                     # receive from previous rank in the specified direction
    )

    """
    print(f'I am rank {cart_rank:2d} at coords {coords} -- '
          f'sending {direction} to {dst} {cart_comm.Get_coords(dst)} :value: {my_buf[0] } '
          f'and receiving from {src} {cart_comm.Get_coords(src)} :value: {nbr_buf[idx]}'
          )
    """

# Compute the 5‐point average: self + four neighbours
total = cart_rank + np.sum(nbr_buf, dtype=np.int64)  
avg_5 = total / 5.0

print(f'After all communication, coords {coords} have neighbours values {nbr_buf} + own {cart_rank}  '
      f'-> average = {avg_5}', flush=True)

# ------------------------------------------------------------------
# 3) Build row- and column- subcommunicators & compute averages
# ------------------------------------------------------------------

# Row communicator: collapse axis 0, keep axis 1
row_comm = cart_comm.Sub([False, True])
# Column communicator: keep axis 0, collapse axis 1
col_comm = cart_comm.Sub([True, False])

# Use numpy arrays for reduction operations for consistency
local_rank = np.array([cart_rank], dtype=np.int32)
row_result = np.empty(1, dtype=np.int32)
col_result = np.empty(1, dtype=np.int32)

row_comm.Allreduce(local_rank, row_result, op=MPI.SUM)
row_avg = row_result[0] / row_comm.Get_size()

col_comm.Allreduce(local_rank, col_result, op=MPI.SUM)
col_avg = col_result[0] / col_comm.Get_size()

# ------------------------------------------------------------------
# 4) Print a summary of the results
# ------------------------------------------------------------------

print(f"[W{world_rank:2d} C{cart_rank:2d} at {coords}] "
                f"nbrs(N,S,E,W)={tuple(int(x) for x in nbr_buf)}  "
                f"avg5={avg_5:.2f}  "
                f"row_avg={row_avg:.2f}  col_avg={col_avg:.2f}", flush=True)


# ------------------------------------------------------------------
# 5) End timing for the entire script
# ------------------------------------------------------------------

end_time = time.time()
execution_time = end_time - start_time

# Reduce to get the maximum time across all processes
max_time = world_comm.reduce(execution_time, op=MPI.MAX, root=0)

if world_rank == 0:
    print(f"\nScript execution time: {max_time:.5f} seconds\n", flush=True)

# ------------------------------------------------------
# 6) Save results to a file
# ------------------------------------------------------

if world_rank == 0:    
    # Prepare data to save
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    # CSV filename
    csv_filename = "mpi_timing_results.csv"
    
    # Check if file exists to determine if we need headers
    file_exists = os.path.exists(csv_filename)
    
    # Open CSV file in append mode
    with open(csv_filename, 'a', newline='') as csvfile:
        fieldnames = ['timestamp', 'processes', 'grid_rows', 'grid_cols', 'execution_time']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        
        # Write header if file is new
        if not file_exists:
            writer.writeheader()
        
        # Write the current run data
        writer.writerow({
            'timestamp': timestamp,
            'processes': world_size,
            'grid_rows': dims[0],
            'grid_cols': dims[1],
            'execution_time': max_time
        })
    
    print(f"Results saved to {csv_filename}", flush=True)
