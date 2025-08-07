#!/bin/bash -l

# conda activate stim39
# Execute this with : sbatch --account none create_subtitles_job.sh

############# SLURM SETTINGS #############
#SBATCH --job-name=subtitles            # some descriptive job name of your choice
#SBATCH --output=%x-%j.out              # output file name will contain job name + job ID
#SBATCH --error=%x-%j.err               # error file name will contain job name + job ID
#SBATCH --time=0-10:00:00               # time limit for the whole run, in the form of d-hh:mm:ss, also accepts mm, mm:ss, hh:mm:ss, d-hh, d-hh:mm
#SBATCH --mem=5G                        # memory required per node, in the form of [num][M|G|T]
#SBATCH --nodes=1                       # number of nodes to allocate, default is 1
#SBATCH --ntasks=1                      # number of Slurm tasks to be launched, increase for multi-process runs ex. MPI
#SBATCH --cpus-per-task=1               # number of processor cores to be assigned for each task, default is 1, increase for multi-threading runs
#SBATCH --ntasks-per-node=1             # number of tasks to be launched on each allocated node
#SBATCH --partition=nodes               # which partition to use, default on MARS is “nodes", or use "nodes" if you want CPU


############# LOADING MODULES (optional) #############

module load apps/anaconda3

############# Anaconda env #############

source activate whisper

python create_subtitles.py
