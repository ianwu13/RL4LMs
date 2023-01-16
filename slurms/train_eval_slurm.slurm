#!/bin/bash
#SBATCH --partition=gpu
#SBATCH --gres=gpu:a100:2
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --time=24:00:00

# Change to proper directory
if [[ "$PWD" =~ "RL4LMs/slurms" ]]
then 
    cd ..
else
    echo 'SLURM JOBS MUST BE RUN FROM "RL4LMs/slurms" FOR THIS PROJECT'
    return -1
fi

# load modules
# module load python/3.7.6
# module load git/2.25.0

# pip3 install torch torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/cu116

# load libraries
# pip install -e .

# setup
module purge
eval "$(conda shell.bash hook)"
conda activate nego_rl

# run
CUDA_LAUNCH_BLOCKING=1 python3 scripts/training/train_text_generation.py --config_path scripts/training/task_configs/nego_generate/ht_nlpo_lr2_tkl5.yml --base_path_to_store_results ../logs/ --experiment_name ht_nlpo_lr2_tkl5