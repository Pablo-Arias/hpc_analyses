#!/bin/sh
############# SLURM SETTINGS #############
#SBATCH --job-name=sync_new_para        # some descriptive job name of your choice
#SBATCH --output=%x-%j.out              # output file name will contain job name + job ID
#SBATCH --error=%x-%j.err               # error file name will contain job name + job ID
#SBATCH --time=2-00:00:00               # time limit for the whole run, in the form of d-hh:mm:ss, also accepts mm, mm:ss, hh:mm:ss, d-hh, d-hh:mm
#SBATCH --mem=15G                       # memory required per node, in the form of [num][M|G|T]
#SBATCH --nodes=1                       # number of nodes to allocate, default is 1
#SBATCH --ntasks=1                      # number of Slurm tasks to be launched, increase for multi-process runs ex. MPI
#SBATCH --cpus-per-task=16              # number of processor cores to be assigned for each task, default is 1, increase for multi-threading runs
#SBATCH --ntasks-per-node=1             # number of tasks to be launched on each allocated node
#SBATCH --partition=nodes               # which partition to use, default on MARS is “nodes", or use "nodes" if you want CPU

############# Anaconda env #############

#Calling with stim39 python
/mnt/autofs/data/userdata/project0028/conda/envs/stim39/bin/python compute_synchrony_parallel.py -u pa121