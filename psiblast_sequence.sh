#!/bin/bash

#SBATCH -J pssm
#SBATCH -p long
#SBATCH -c 4                            # one CPU core
#SBATCH -t 7-30:30:00
#SBATCH --mem=10G 
#SBATCH -o pssm-slurm.out #%j.out
#SBATCH --mail-type=ALL
#SBATCH --mail-user=ndtrvlmind@gmail.com


source /home/nn121/.bash_profile
source activate pconda37

export DIR_BASE=/n/groups/alquraishi/Apps/databases
export PATH=${DIR_BASE}:$PATH
export DIR_SCRIPTS=${DIR_BASE}/scripts
#export DB=${DIR_BASE}/tool_databases/blast/uniparc/casp12
export DB=/n/scratch3/users/n/nn121/uniprot_sprot
export DIR_INPUT=./20K_input_fasta
export DIR_OUTPUT=./20K_out_pssm
export NUM_SAMPLES=12
export NUM_THREADS=8
#export FILES="${@:1}"    #accepting file input

unset IFS
for FILE in ${DIR_INPUT}/*.fasta; do
    echo $(basename ${FILE})
	rm -f ${DIR_OUTPUT}/${FILE}.out
	#psiblast -db $DB -query ${FILE} -num_iterations 3 -inclusion_ethresh 0.001 -out_ascii_pssm ${DIR_OUTPUT}/$(basename ${FILE}).pssm -out ${DIR_OUTPUT}/$(basename ${FILE}).out -num_threads ${NUM_THREADS}
    #only need pssm files
    psiblast -db $DB -query ${FILE} -num_iterations 3 -inclusion_ethresh 0.001 -out_ascii_pssm ${DIR_OUTPUT}/$(basename ${FILE}).pssm -num_threads ${NUM_THREADS}


done
