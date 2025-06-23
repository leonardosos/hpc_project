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
# 2 B) Exchange ranks with N, S, E, W neighbours and compute the 5‐point average
# ------------------------------------------------------------------

# Pre-allocated arrays 
nbr_buf = np.empty(4, dtype=np.int32) # Buffer for neighbours: [north, south, east, west]
my_buf  = np.array([cart_rank], dtype=np.int32) # Buffer for self rank

# Exchange with NORTH neighbor (axis=0, displacement=-1)
# This communicates with the rank above in the grid
src_north, dst_north = cart_comm.Shift(0, -1)  # Get north neighbor ranks
cart_comm.Sendrecv(
    sendbuf=my_buf[0:1],        # Send my cart_rank to north neighbor
    dest=dst_north,             # Destination: north neighbor
    recvbuf=nbr_buf[0:1],       # Receive north neighbor's rank into index 0
    source=src_north            # Source: north neighbor
)

# Exchange with SOUTH neighbor (axis=0, displacement=+1)  
# This communicates with the rank below in the grid
src_south, dst_south = cart_comm.Shift(0, +1)  # Get south neighbor ranks
cart_comm.Sendrecv(
    sendbuf=my_buf[0:1],        # Send my cart_rank to south neighbor
    dest=dst_south,             # Destination: south neighbor
    recvbuf=nbr_buf[1:2],       # Receive south neighbor's rank into index 1
    source=src_south            # Source: south neighbor
)

# Exchange with EAST neighbor (axis=1, displacement=+1)
# This communicates with the rank to the right in the grid
src_east, dst_east = cart_comm.Shift(1, +1)    # Get east neighbor ranks
cart_comm.Sendrecv(
    sendbuf=my_buf[0:1],        # Send my cart_rank to east neighbor
    dest=dst_east,              # Destination: east neighbor
    recvbuf=nbr_buf[2:3],       # Receive east neighbor's rank into index 2
    source=src_east             # Source: east neighbor
)

# Exchange with WEST neighbor (axis=1, displacement=-1)
# This communicates with the rank to the left in the grid
src_west, dst_west = cart_comm.Shift(1, -1)    # Get west neighbor ranks
cart_comm.Sendrecv(
    sendbuf=my_buf[0:1],        # Send my cart_rank to west neighbor
    dest=dst_west,              # Destination: west neighbor
    recvbuf=nbr_buf[3:4],       # Receive west neighbor's rank into index 3
    source=src_west             # Source: west neighbor
)

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

# Reduce to get time for all processes
total_time = world_comm.reduce(execution_time, op=MPI.SUM, root=0)

if world_rank == 0:
    print(f"\nScript execution time: {total_time:.5f} seconds\n", flush=True)

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
            'execution_time': f"{total_time:.5f}"
        })
    
    print(f"Results saved to {csv_filename}", flush=True)
