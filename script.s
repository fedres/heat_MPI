#!/bin/sh

#SBATCH --partition=private-cui-cpu
#SBATCH --ntasks=100
#SBATCH --cpus-per-task=1

#SBATCH --job-name=mpi_scaling_test
#SBATCH --output=slurm-mpi.out
#SBATCH --mail-user=thyagarajan.karthik@unige.ch
#SBATCH --mail-type=FAIL
#SBATCH --mail-type=ALL
#SBATCH --time=7-00:00:00

module load foss

srun ./run
