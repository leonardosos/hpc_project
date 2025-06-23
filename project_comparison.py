"""
Performance Comparison: MPI Neighbor Communication in 2D Cartesian Grid

This script benchmarks two different approaches for exchanging rank information 
between neighboring processes in a 2D Cartesian MPI topology:

Approach 2A (Loop-based): 
    - Uses a dictionary-driven for loop to iterate through all four neighbors
    - Single loop handles north, south, east, and west communications sequentially
    - More compact and maintainable code structure

Approach 2B (Explicit calls):
    - Uses separate, explicit MPI_Sendrecv calls for each directional neighbor
    - Four distinct communication calls: north → south → east → west
    - More verbose but potentially clearer control flow

Both approaches:
    - Create a 2D periodic Cartesian communicator from MPI_COMM_WORLD
    - Exchange rank values with immediate neighbors (N, S, E, W)
    - Compute a 5-point stencil average (self + 4 neighbors)
    - Measure and compare execution times

The script helps determine if loop-based neighbor communication introduces
any performance overhead compared to explicit calls in MPI applications.

RESULTS:

Script execution time: 0.00058 seconds (1A)

Script execution time: 0.00020 seconds (2B)

"""



# basic library imports
from mpi4py import MPI
import numpy as np
import time

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

# Start timing the entire script
start_time_1 = time.time()

# ------------------------------------------------------------------
# 2 A) Exchange ranks with N, S, E, W neighbours and compute the 5‐point average
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
    
# Compute the 5‐point average: self + four neighbours
total = cart_rank + np.sum(nbr_buf, dtype=np.int64)  
avg_5 = total / 5.0

print(f'After all communication, coords {coords} have neighbours values {nbr_buf} + own {cart_rank}  '
      f'-> average = {avg_5}', flush=True)

end_time_1 = time.time()
total_time_1 = end_time_1 - start_time_1

start_time_2 = time.time()


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
 
end_time_2 = time.time()
total_time_2 = end_time_2 - start_time_2


# ------------------------------------------------------------------
# 3) Report the execution time
# ------------------------------------------------------------------

# Reduce to get the maximum time across all processes
total_time_1 = world_comm.reduce(total_time_1, op=MPI.MAX, root=0)
total_time_2 = world_comm.reduce(total_time_2, op=MPI.MAX, root=0)


if world_rank == 0:
    print(f"\nScript execution time: {total_time_1:.5f} seconds (1A)\n", flush=True)
    print(f"\nScript execution time: {total_time_2:.5f} seconds (2B)\n", flush=True)