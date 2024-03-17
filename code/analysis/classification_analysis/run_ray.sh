#!/bin/bash

#SBATCH --job-name=abcd
#SBATCH --partition=amilan
#SBATCH --ntasks-per-node=32
#SBATCH --mem-per-cpu=3500

### --job-name=abcd
### --account=ucb-general
### --partition=aa100
### --ntasks=4
### --cpus-per-task=24
### --gres=gpu


source ~/.bashrc

source /pl/active/banich/studies/Relevantstudies/abcd/env/bin/activate



ray start --head --port=8888
ray start --address='198.59.51.5:8888'

python3 /pl/active/banich/studies/Relevantstudies/abcd/data/clustering/analysis/classification_analysis/run_ray.py