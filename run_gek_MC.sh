#!/bin/bash
#SBATCH --job-name=gek_exp_MC
#SBATCH --output=logs_MC/%x_%j.out
#SBATCH --error=logs_MC/%x_%j.err
#SBATCH --time=03:00:00
#SBATCH --cpus-per-task=1
#SBATCH --mem=1G

source ~/.bashrc
conda activate GEK

python main.py \
    --d $1 \
    --h $2 \
    --alpha $3 \
    --seed $4 \
    --var_threshold $5 \
    --length_scale $6 \
    --sigma $7 \
    --inner_lr $8 \
    --method $9 \
    --parallel ${10} \
    --T ${11} \
    --max_iter ${12}
