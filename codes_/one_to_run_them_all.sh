#!/bin/bash
#SBATCH --output=slurm/%x_%j.out 
#SBATCH --error=slurm/%x_%j.err 
#SBATCH --gres=gpu:4 
#SBATCH --time=48:00:00 
#SBATCH --nodes=1 
#SBATCH --mail-type=END,FAIL 
#SBATCH --mail-user=f.assmuth@fz-juelich.de

srun run_all_depth_test.sh
srun run_all_depth.sh
srun run_all_drop.sh
srun run_all_early_stop.sh
srun run_all_fig2.sh
srun run_all_fig3.sh
srun run_all_fig6B.sh
srun run_all_fig6D.sh
srun run_all_lr.sh
srun run_analysis_eval_depth.sh
srun run_analysis_eval_dropout.sh
srun run_analysis_eval_learn_rate.sh
srun run_analysis_eval.sh