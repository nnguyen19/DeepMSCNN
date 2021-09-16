#!/bin/bash
#SBATCH -J train_CNN
#SBATCH -p medium
#SBATCH -c 4
#SBATCH -t 30:00:00
#SBATCH -p gpu_quad
#SBATCH --gres=gpu:2
#SBATCH --mem=10G 
#SBATCH -o train_pssm_slurm.out
#SBATCH --mail-type=ALL
#SBATCH --mail-user=nnguyen19@ole.augie.edu
 
module load gcc/6.2.0
module load cuda/9.0

source /home/nn121/.bash_profile
source activate pconda37
 
python3 train_CNN_avgfp.py