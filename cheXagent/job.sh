#!/bin/bash                      
#SBATCH -n 1                     
#SBATCH --time=15:00:00
#SBATCH --gres=gpu:a100:1
#SBATCH --mem=20G
module load openmind8/apptainer/1.1.7
module show openmind8/cuda/12.1
apptainer exec --nv /om2/user/sophiejg/cxr-container python /om2/user/sophiejg/mlhc_rology/cheXagent/gen_img_reports.py  # Run a program using GPU (e.g. a Python program). 
