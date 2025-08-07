#!/bin/sh
############# SLURM SETTINGS #############
#SBATCH --job-name=hr_synnchrony        # some descriptive job name of your choice
#SBATCH --output=%x-%j.out              # output file name will contain job name + job ID
#SBATCH --error=%x-%j.err               # error file name will contain job name + job ID
#SBATCH --time=5-00:00:00                # time limit for the whole run, in the form of d-hh:mm:ss, also accepts mm, mm:ss, hh:mm:ss, d-hh, d-hh:mm
#SBATCH --mem=80G                       # memory required per node, in the form of [num][M|G|T]
#SBATCH --nodes=1                       # number of nodes to allocate, default is 1
#SBATCH --ntasks=1                      # number of Slurm tasks to be launched, increase for multi-process runs ex. MPI
#SBATCH --cpus-per-task=60              # number of processor cores to be assigned for each task, default is 1, increase for multi-threading runs
#SBATCH --ntasks-per-node=1             # number of tasks to be launched on each allocated node
#SBATCH --partition=nodes               # which partition to use, default on MARS is “nodes", or use "nodes" if you want CPU


############# LOADING MODULES (optional) #############

module load apps/anaconda3

module load apps/ffmpeg/6.0.0/gcc-4.8.5+x264


############# Anaconda env #############

conda activate stim39

python all_synchrony.py -u pa121