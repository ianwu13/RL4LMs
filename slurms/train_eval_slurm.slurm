#!/bin/bash
#SBATCH --partition=gpu
#SBATCH --gres=gpu:a100:2
#SBATCH --ntasks=1
#SBATCH --mem=16GB
#SBATCH --cpus-per-task=16
#SBATCH --time=10:15:00

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
python3 scripts/training/train_text_generation.py --config_path scripts/training/task_configs/nego_generate/t5_nlpo_on_supervised.yml --base_path_to_store_results ../logs/ --experiment_name nego_gen_nlpo_pollution2_food_coef0_usesubstr_newv

# configs
# nego_gen_nlpo_meteor1_lowkl_vhigh_lr
# scripts/training/task_configs/synthetic_generate_increasing_numbers/t5_nlpo.yml
# scripts/training/task_configs/nego_target/t5_supervised_combined.yml
# scripts/training/task_configs/dialog/gpt2_nlpo_on_supervised.yml
# scripts\training\task_configs\casino\t5_supervised.yml
# nego_target_sup_comb_lr1e4_rw_k4_co_l256_rand_nopers_accs