#!/bin/bash
#SBATCH --partition=gpu
#SBATCH --gres=gpu:v100:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --time=05:15:00

# Change to proper directory
if [[ "$PWD" =~ "RL4LMs/slurms" ]]
then 
    cd ..
else
    echo 'SLURM JOBS MUST BE RUN FROM "RL4LMs/slurms" FOR THIS PROJECT'
    return -1
fi

# load modules
module load python/3.7.6

# load libraries
pip install -e .

# run
python3 scripts/training/train_text_generation.py --config_path scripts/training/task_configs/synthetic_generate_increasing_numbers/t5_nlpo.yml --base_path_to_store_results ../logs/