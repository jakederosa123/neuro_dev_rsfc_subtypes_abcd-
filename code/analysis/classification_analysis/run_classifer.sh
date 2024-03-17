#!/bin/bash

#SBATCH --job-name=subtyping
#SBATCH --qos=blanca-ics
#SBATCH --partition=blanca-ics
#SBATCH --account=blanca-ics-clearmem
#SBATCH --cpus-per-task=1
#SBATCH --export=None

source /curc/sw/anaconda3/latest

conda activate jake

python3 /pl/active/banich/studies/abcd/data/clustering/analysis/classification.py