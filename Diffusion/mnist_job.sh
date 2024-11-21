#!/bin/bash
#BSUB -J mnist_training           # Job name
#BSUB -q gpua10                  # Queue name
#BSUB -gpu "num=1"  # Request 1 GPU
#BSUB -n 4                        # Request 4 CPU cores
#BSUB -R "span[hosts=1]"          # Ensure resources are on a single node
#BSUB -W 02:00                    # Set walltime (2 hours)
#BSUB -R "rusage[mem=4096]"       # Request 4GB of system memory
#BSUB -o output_%J.log            # Output file
#BSUB -e error_%J.log             # Error file

# Load modules
module load python3/3.10.12
module load cuda/12.1

# Set W&B API key (replace with your actual key)
export WANDB_API_KEY="6ecda4c80f57815b4ff4780014d596e19617454c"


# Run the training script
python ddpm.py
