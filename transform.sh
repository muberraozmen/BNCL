#!/bin/bash
#SBATCH --nodes=1                   # Number of nodes
#SBATCH --mem=16G                    # Memory (per node)
#SBATCH --gres=gpu:1                # Number of GPUs (per node)
#SBATCH --time=0-48:00
#SBATCH --output=out.log            # output file
#SBATCH --error=err.log             # error file

python3.8 -u transformation/converter.py