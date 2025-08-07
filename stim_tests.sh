#!/bin/bash -l

# conda activate stim39
# Execute this with : sbatch --account project0028 stim_tests.sh

############# SLURM SETTINGS #############
#SBATCH --job-name=stim_test            # some descriptive job name of your choice
#SBATCH --output=%x-%j.out              # output file name will contain job name + job ID
#SBATCH --error=%x-%j.err               # error file name will contain job name + job ID
#SBATCH --time=0-0:00:30                # time limit for the whole run, in the form of d-hh:mm:ss, also accepts mm, mm:ss, hh:mm:ss, d-hh, d-hh:mm
#SBATCH --mem=5G                        # memory required per node, in the form of [num][M|G|T]
#SBATCH --nodes=1                       # number of nodes to allocate, default is 1
#SBATCH --ntasks=1                      # number of Slurm tasks to be launched, increase for multi-process runs ex. MPI
#SBATCH --cpus-per-task=1               # number of processor cores to be assigned for each task, default is 1, increase for multi-threading runs
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

source activate stim39

cd /users/pa121h/development/STIM/tests

python -m unittest discover .