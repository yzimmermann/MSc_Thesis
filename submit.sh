#!/bin/bash
#SBATCH --job-name=gek_exp
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
#SBATCH --time=00:59:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=6G

source ~/.bashrc
conda activate GEK

python main_deterministic.py \
    --d $1 \
    --h $2 \
    --alpha $3 \
    --seed $4 \
    --var_threshold $5 \
    --length_scale $6 \
    --sigma $7 \
    --sigma_f $8 \
    --sigma_g $9 \
    --inner_lr ${10} \
    --method ${11}
