#!/bin/bash

# Example of running python script in a batch mode

#SBATCH -J feature_stacking
#SBATCH -p medium
#SBATCH -c 4                            # one CPU core
#SBATCH -t 2-24:30
#SBATCH --mem=10G 
#SBATCH -o feature_stack-%j.out
#SBATCH --mail-type=ALL
#SBATCH --mail-user=nnguyen19@ole.augie.edu


source /home/nn121/.bash_profile
source activate pconda37

python3 feature_encoding.py
