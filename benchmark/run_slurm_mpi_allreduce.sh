#!/bin/sh -l
#SBATCH --cpus-per-task=128
#SBATCH -t 60

source /home/users/ppochelu/project/bigdata_tuto/jax/env.sh

NUM_RANKS=$(($SLURM_JOB_NUM_NODES * 128))

echo "NUM_RANKS=${NUM_RANKS}"

mpirun --oversubscribe -n $NUM_RANKS python3 ./benchmark/MPI_allreduce.py
