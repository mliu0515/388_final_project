#!/bin/bash
#SBATCH -J run           # Job name
#SBATCH -o run.o%j       # Name of stdout output file
#SBATCH -e run.e%j       # Name of stderr error file
#SBATCH -p gpu-a100-small        # Queue (partition) name
#SBATCH -N 1               # Total # of nodes (must be 1 for serial)
#SBATCH -n 1               # Total # of mpi tasks (should be 1 for serial)
#SBATCH -t 00:30:00        # Run time (hh:mm:ss)
#SBATCH -A CCR24006        # Project/Allocation name (req'd if you have more than 1)
cd /work/07016/cw38637/ls6/nlp/

python video_analyzer.py

# torchrun --nproc_per_node 1 llama3_example.py
