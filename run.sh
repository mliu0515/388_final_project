#!/bin/bash
#SBATCH -J run           # Job name
#SBATCH -o run.o%j       # Name of stdout output file
#SBATCH -e run.e%j       # Name of stderr error file
#SBATCH -p vm-small        # Queue (partition) name
#SBATCH -N 1               # Total # of nodes (must be 1 for serial)
#SBATCH -n 1               # Total # of mpi tasks (should be 1 for serial)
#SBATCH -t 1:00:00        # Run time (hh:mm:ss)
#SBATCH -A EAR23036        # Project/Allocation name (req'd if you have more than 1)
cd /work/07016/cw38637/ls6/nlp/

# module load cuda/12.0
source activate

python blip2.py
