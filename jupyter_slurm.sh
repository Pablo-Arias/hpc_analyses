#!/bin/bash

############# SLURM SETTINGS #############
#SBATCH --account=none
#SBATCH --job-name=Jupyter-Notebook
#SBATCH --partition=nodes
#SBATCH --time=0-08:00:00
#SBATCH --mem=60G
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --ntasks-per-node=1

############# LOADING MODULES (optional) #############
module load apps/miniforge

#Start jupyter server
jupyter server --no-browser --port=12365 --ip 0.0.0.0