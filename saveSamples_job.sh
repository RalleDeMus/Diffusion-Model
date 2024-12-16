# mnist_job.sh
#!/bin/bash
#BSUB -J mnist_training           # Job name
#BSUB -q gpua100                   # Queue name for Tesla A10 GPUs
#BSUB -gpu "num=1"                # Request 1 GPU in exclusive mode
#BSUB -n 4                        # Request 4 CPU cores (required)
#BSUB -R "span[hosts=1]"          # Ensure resources are on a single node
#BSUB -W 72:00                    # Walltime (1 hour)
#BSUB -R "rusage[mem=4096]"       # Request 4GB of system memory
#BSUB -o ./logs/outputLogs/output_%J.log            # Output file
#BSUB -e ./logs/errorLogs/error_%J.log             # Error file

# Load necessary modules
module load python3/3.10.12
module load cuda/12.1

# Set W&B API key (replace with your actual key)
export WANDB_API_KEY="6ecda4c80f57815b4ff4780014d596e19617454c"

# Activate virtual environment
source /zhome/1a/a/156609/project/path/.venv/bin/activate

# Run the PyTorch training script
python3 generateAndRead/save_model_binary.py 
