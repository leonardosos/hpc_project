#!/bin/bash -x
#
#PBS -N DAE  ! job name
#PBS -j oe   !merge std-err and std-out
#PBS -q s3par !queue
#PBS -l nodes=1:ppn=16 # max 16 task, max 1 node
#PBS -l walltime=00:10:00

module purge
module load oneapi/compiler
module load oneapi/mkl
module load oneapi/mpi
module load IMPI/west5.4.0

MYHOME="/home/corsohpc2/project_Brighenti/hpc_project"
DIR_JOB=${MYHOME}
cd $DIR_JOB
mpirun -np 10 python3 project.py > output_10.out