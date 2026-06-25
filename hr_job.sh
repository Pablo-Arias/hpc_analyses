#!/bin/bash -l

############# SLURM SETTINGS #############
#SBATCH --job-name=hr_analysis          # some descriptive job name of your choice
#SBATCH --output=%x-%j.out              # output file name will contain job name + job ID
#SBATCH --error=%x-%j.err               # error file name will contain job name + job ID
#SBATCH --time=7-0:00:00                # time limit for the whole run, in the form of d-hh:mm:ss, also accepts mm, mm:ss, hh:mm:ss, d-hh, d-hh:mm
#SBATCH --mem=10G                       # memory required per node, in the form of [num][M|G|T]
#SBATCH --nodes=1                       # number of nodes to allocate, default is 1
#SBATCH --ntasks=1                      # number of Slurm tasks to be launched, increase for multi-process runs ex. MPI
#SBATCH --cpus-per-task=2               # number of processor cores to be assigned for each task, default is 1, increase for multi-threading runs
#SBATCH --ntasks-per-node=1             # number of tasks to be launched on each allocated node

## -- GPU
#SBATCH --partition=nodes                 # which partition to use, default on MARS is “nodes", or use gpu if you want GPU
##SBATCH --gres=gpu:1                    # Ask to get GPU allocated, do this if using gpu

############# LOADING MODULES (optional) #############

module load apps/apptainer/1.3.4

############# Anaconda env #############

singularity exec --cleanenv --nv --no-home -B "$(pwd)":/mnt -B /mnt/data/project0028/repos/STIM:/repos/STIM/ -B /mnt/data/project0028/repos/video_analysis/:/repos/video_analysis/ -B /mnt/data/project0028/repos/pyVHR/:/repos/pyVHR/ -B "$(pwd)/aux_folder":/users --env PATH=/opt/miniconda3/bin:$PATH interalysis0.1.sif /bin/bash -c "source activate pyvhr && python /mnt/hr.py"