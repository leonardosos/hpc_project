# HPC Project: 2D Cartesian Grid Topology with MPI

## Project Description

Write a parallel code (using both **Python** and **Fortran**).

Create a 2-dimensional Cartesian grid topology to communicate between processes. Each task initializes a variable with the local rank of the Cartesian communicator.

## Exercise Steps

The exercise is divided into three steps:

1. **Compare ranks**: Compare the local rank with the global `MPI_COMM_WORLD` rank. Are they the same number?

2. **Calculate neighbor averages**: Calculate on each task the average between its local rank and the local rank for each of its neighbours (north, east, south, west).
   > **Note**: In order to do this the Cartesian communicator has to be periodic (the bottom rank is a neighbour of the top).

3. **Row and column averages**: Calculate the average of the local ranks on each row and column. Create a family of sub-cartesian communicators to allow the communications between rows and columns.

## Requirements

- The code must include **comments** that explain how the procedure was implemented
- In the report, in addition to an introduction to the problem, there must be an adequate description of how the algorithm was implemented
- The **efficiency** of the code must be shown and discussed

## Deadlines

- **Draft submission**: Friday, June 27 (evening)
  - The committee will take a few days to evaluate the project and possibly suggest changes
- **Final submission**: July 2 at 10:30 AM

## Useful MPI Functions (Fortran)

| Function | Signature |
|----------|-----------|
| `MPI_DIMS_CREATE` | `int MPI_Dims_create(int nnodes, int ndims, int *dims)` |
| `MPI_CART_CREATE` | `int MPI_Cart_create(MPI_Comm comm_old, int ndims, int *dims, int *periods, int reorder, MPI_Comm *comm_cart)` |
| `MPI_CART_COORDS` | `int MPI_Cart_coords(MPI_Comm comm, int rank, int maxdims, int *coords)` |
| `MPI_CART_SHIFT` | `int MPI_Cart_shift(MPI_Comm comm, int direction, int disp, int *rank_source, int *rank_dest)` |
| `MPI_CART_SUB` | `int MPI_Cart_sub(MPI_Comm comm, int *remain_dims, MPI_Comm *newcomm)` |
| `MPI_COMM_FREE` | `int MPI_Comm_free(MPI_Comm *comm)` |
| `MPI_SENDRECV` | `int MPI_Sendrecv(void *sendbuf, int sendcount, MPI_Datatype sendtype, int dest, int sendtag, void *recvbuf, int recvcount, MPI_Datatype recvtype, int source, int recvtag, MPI_Comm comm, MPI_Status *status)` |
| `MPI_REDUCE` | `int MPI_Reduce(void* sendbuf, void* recvbuf, int count, MPI_Datatype datatype, MPI_Op op, int root, MPI_Comm comm)` |
| `MPI_INIT` | `int MPI_Init(int *argc, char ***argv)` |
| `MPI_COMM_SIZE` | `int MPI_Comm_size(MPI_Comm comm, int *size)` |
| `MPI_COMM_RANK` | `int MPI_Comm_rank(MPI_Comm comm, int *rank)` |
| `MPI_FINALIZE` | `int MPI_Finalize(void)` |
