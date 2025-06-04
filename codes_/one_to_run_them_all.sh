#!/bin/bash
#SBATCH --array=0-8
#SBATCH --output=slurm/%x_%j_$A.out 
#SBATCH --error=slurm/%x_%j_$A.err 
#SBATCH --gres=gpu:1
#SBATCH --time=48:00:00 
#SBATCH --nodelist=pgi15-cpu2
#SBATCH --nodes=1
#SBATCH --mail-type=END,FAIL 
#SBATCH --mail-user=f.assmuth@fz-juelich.de
#SBATCH --job-name=dANN

SCRIPT=$(sed -n "$((SLURM_ARRAY_TASK_ID+1))p" joblist.txt)

uv sync
sh "$SCRIPT" all_out > /dev/null