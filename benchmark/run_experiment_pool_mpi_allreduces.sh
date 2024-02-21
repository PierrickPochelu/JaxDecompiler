#!/bin/sh

export PYTHONPATH="${PYTHONPATH}:${PWD}"

sbatch --nodes=1 ./benchmark/run_slurm_mpi_allreduce.sh
sbatch --nodes=4 ./benchmark/run_slurm_mpi_allreduce.sh
sbatch --nodes=16 ./benchmark/run_slurm_mpi_allreduce.sh
sbatch --nodes=64 ./benchmark/run_slurm_mpi_allreduce.sh

