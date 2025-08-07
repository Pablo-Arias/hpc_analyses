#!/bin/bash -l

############# SLURM SETTINGS #############
#SBATCH --job-name=process_video        # some descriptive job name of your choice
#SBATCH --output=%x-%j.out              # output file name will contain job name + job ID
#SBATCH --error=%x-%j.err               # error file name will contain job name + job ID
#SBATCH --time=1-0:00:00                # time limit for the whole run, in the form of d-hh:mm:ss, also accepts mm, mm:ss, hh:mm:ss, d-hh, d-hh:mm
#SBATCH --mem=20G                       # memory required per node, in the form of [num][M|G|T]
#SBATCH --nodes=1                       # number of nodes to allocate, default is 1
#SBATCH --ntasks=1                      # number of Slurm tasks to be launched, increase for multi-process runs ex. MPI
#SBATCH --cpus-per-task=40              # number of processor cores to be assigned for each task, default is 1, increase for multi-threading runs
#SBATCH --ntasks-per-node=1             # number of tasks to be launched on each allocated node
#SBATCH --partition=nodes               # which partition to use, default on MARS is “nodes", or use "nodes" if you want CPU


############# LOADING MODULES (optional) #############

module load apps/miniforge

# For ffmpeg local instalation
export INSTALL_DIR="/mnt/data/project0028"
export SRC_DIR="$INSTALL_DIR/ffmpeg_sources"
export PATH="$INSTALL_DIR/bin:$PATH"
export PKG_CONFIG_PATH="$INSTALL_DIR/lib/pkgconfig"


############# Anaconda env #############

conda activate stim39

python process_videos.py -u pa121h

