#!/bin/bash
#SBATCH --job-name=hrws_diffusion
#SBATCH --output=%x_%j_out.txt
#SBATCH --error=%x_%j_err.txt
# NOT SBATCH --gpus=a100:8
#SBATCH --gpus=8
#SBATCH --cpus-per-task=2
#SBATCH --ntasks-per-node=8
#SBATCH --nodes=1
#SBATCH --mem=32G
#SBATCH --time=72:15:00


# Echo time and hostname into log
echo "Date:     $(date)"
echo "Hostname: $(hostname)"


module --quiet purge

module load anaconda/3

conda activate /home/mila/m/mohit.anand/miniforge3/envs/forest_mssl

cd /home/mila/m/mohit.anand/projects/earthsphere-diffusion/

CUDA_LAUNCH_BLOCKING=1
# Execute Python script

srun python train_ElucidateDiffusion.py