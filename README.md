# Introduction

Here are some scripts to streamline the process of analysing data collected with DuckSoup
Follow these steps to get ready:
- These different scripts use some of our repos to analyse video recordings.

## Preprocessing of videos
- Put your original video inside: original_data/name_of_experiment/
Then try and run the process_videos.py script by doing the folllowing:

First, create a new conda environment with requirements:
```
conda create --name ds_prepro python=3.9
conda activate ds_prepro
pip install soundfile pandas opencv-python
```

Second, install dependencies such as ffmpeg.
```
brew install ffmpeg
```

Third, clone the repository, as well as the one containing some dependencies:
```
git clone https://github.com/ducksouplab/prepro.git
git clone https://github.com/Pablo-Arias/STIM.git 
```

Add these to your path—see below if you have an error with these commands:
```
conda develop "$(pwd)/STIM"
conda develop "$(pwd)/prepro"
```

Sometimes the conda develop is not recognised by the system. In that case, just put these lines of code at the begining of your script:
```
import sys
from pathlib import Path

root = Path(__file__).resolve().parent   # directory that contains the script, where the repositories are located, change as required
sys.path.append(str(root / "STIM"))
sys.path.append(str(root / "prepro"))

# now you can do:
import stim          # whatever the package's top-level name is
import prepro
```

Now you put the raw data collected with ducksoup in a folder called "original_data/name_of_experiment/ and execute the process_videos.py script. If you are in an HPC using SLURM, you can use :
sbatch --account project_nb process_audio_job.sh

Note that you have to change "project_nb" to your actual project number

But checkout that process_videos_job.sh is using the parameters you need.

## Face analysis
Use the AU_analysis_mp.py script and AU_analysis_mp_job.py files

## Voice analysis
Use the voice_analysis.py script and associated "_job" file.
 
## Transcription
Use the transcription.py file

## For heart rate
you will need to download the file interalysis0.1.sif


