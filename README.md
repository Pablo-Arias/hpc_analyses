# Hpc_analyses — Setting up Environments, Scripts and SLURM Jobs

This repository contains **small, composable pipelines** to preprocess and analyse **DuckSoup** experiment data on laptops and **SLURM** clusters. The goal is to make routine operations (video prep, facial AUs, voice features, transcription→subtitles, heart-rate) **repeatable and scalable**.

- Each script does **one job** (preprocess, AU extraction, audio prep, voice analysis, transcription, subtitles, Heart Rate).
- **Environment instructions** (which conda env + `conda develop` lines) are written **at the top of each `.py` file**, and matched by the associated `*_job.sh` wrapper.
- This README mirrors those **per‑script environments** so new users can get running quickly.

> Companion repos used by these pipelines:
>
> - **prepro** – helper functions for DuckSoup preprocessing: https://github.com/ducksouplab/prepro  
> - **STIM** – AV transformation + audio features: https://github.com/Pablo-Arias/STIM  
> - **video_analysis** – wrappers/tutorials for face/HR/transcribe: https://github.com/Pablo-Arias/video_analysis  

---

## What each pipeline is for (high‑level)

- **Video preprocessing (`process_videos.py`)**: standardize screen/camera recordings, resample, fix mux/layout for downstream steps.
- **Facial Action Units (MediaPipe & optional GPU)**:
  - `AU_analysis_mp.py`: **CPU** MediaPipe face features → AU time series.
  - `AU_analysis.py` / `AU_analysis_GPU.py`: **alternative/GPU** AU pipelines (if your cluster supports GPU).
  - `create_au_videos.py`: overlay AU results back onto videos for QA.
- **Audio preprocessing (`process_audio.py`)**: extract WAV, resample/normalize.
- **Voice analysis (`voice_analysis.py`)**: compute acoustic features via **STIM** (pitch, loudness, etc.) for stats.
- **Transcription → Subtitles (`transcription.py`, `create_subtitles.py`)**: ASR (e.g., Whisper variants) → clean text → VTT/SRT.
- **Heart‑rate extraction (`hr.py`)**: derive HR from video; some setups call a container (e.g., `.sif`).

All scripts accept `--help` to show their CLI flags (paths, filters, etc.).

---

# 1) Miniforge/conda (one‑time)

If not installed, set up **Miniforge** from conda‑forge and initialize your shell so `conda activate` works:

```bash
source ~/miniforge3/etc/profile.d/conda.sh
conda init bash     # or: zsh/fish
# open a new shell after 'conda init' (or: source ~/.bashrc)
```

**ffmpeg** (media I/O) is recommended **inside** the envs or system‑wide:
```bash
# inside any env
conda install -y -c conda-forge ffmpeg
# or system-wide (macOS): brew install ffmpeg
```

---

# 2) Clone helper repos *next to* this repo

From the **parent directory** of your `hpc_analyses` checkout:

```bash
git clone https://github.com/ducksouplab/prepro
git clone https://github.com/Pablo-Arias/STIM
# optional extras/wrappers:
git clone https://github.com/Pablo-Arias/video_analysis
```

Your tree should look like:
```
parent_dir/
├─ hpc_analyses/
├─ STIM/
└─ prepro/
└─ video_analysis/
```

---

# 3) Environments overview (do once)

Below are the **core environments** used across the scripts that you will use. If the conda develop command doesn't work, see note below.

## A) `ds_prepro` environment 
ds_prepro stands for duscksoup preprocessing, which handles all video preprocessing, MediaPipe AUs, basic audio

Create the environment as follows:
```bash
conda create -y -n ds_prepro python=3.9 -c conda-forge
conda activate ds_prepro
pip install pandas soundfile opencv-python mediapipe tqdm
conda install -y -c conda-forge ffmpeg conda-build
conda develop ../repos/STIM
conda develop ../repos/video_analysis
```

If you don't manage to install ffmpeg, consider installing it locally follwing [this tutorial](https://github.com/ducksouplab/prepro/blob/main/tutorial/build_ffmpeg_locally_for_HPC.md)

## B) `stim39` — advanced audio/voice analysis (via STIM)

Stim39 is a repo to perform voice analyses.

```bash
conda create -y -n stim39 python=3.9 -c conda-forge
conda activate stim39
pip install numpy scipy pandas matplotlib soundfile pyloudnorm             opencv-python praat-parselmouth pyo
# optional (some STIM features):
conda install -y -c roebel easdif || true
conda install -y -c conda-forge ffmpeg conda-build
conda develop ../repos/STIM
conda develop ../repos/video_analysis
```

---

## C) `whisper` — OpenAI Whisper ASR *(PyTorch backend)*

> Use if your `transcription.py`/job files indicate **OpenAI Whisper**.

```bash
conda create --name whisper python=3.11
conda activate whisper
pip install whisper
pip3 install whisper-timestamped
pip install opencv-python
pip install auditok
conda develop ../repos/STIM
conda develop ../repos/video_analysis
```


# 4) Make `STIM` & `prepro` importable (per env)

You may be able to use conda develop to add the path of your repositories to your scripts. 
Do this **once per env** you plan to use:

```bash
# In ds_prepro (and ds_prepro-gpu, whisper, fasterwhisper if needed)
conda activate ds_prepro
conda develop "$(pwd)/../STIM"
conda develop "$(pwd)/../prepro"

# In stim39
conda activate stim39
conda develop "$(pwd)/../STIM"
conda develop "$(pwd)/../prepro"
```

If conda develop doesn't work, you can add this snippet at the top of your scripts:
>
> ```python
> from pathlib import Path; import sys
> root = Path(__file__).resolve().parent
> sys.path += [str(root.parent / "STIM"), str(root.parent / "prepro")]
> ```

---

# 5) SLURM: use the job scripts already in this repo

Use the provided `_job.sh` wrappers **as‑is**. Typically you only edit:
- `#SBATCH --account=…` (your project)
- `#SBATCH --partition=…` (queue/partition)
- time/mem/CPU (and `--gres=gpu:…` if GPU)
- the **Miniforge activation** (two options below)

**Submit & monitor**
```bash
mkdir -p logs
sbatch --account <YOUR_ACCOUNT> process_videos_job.sh
squeue -u "$USER"
tail -f logs/<jobname>-<jobid>.out
seff <JOBID>
sstat -j <JOBID> --format=JobID,MaxRSS,AveRSS,MaxVMSize
```


## Preprocessing videos:

Open the process_videos.py and corresponding process_videos_job.py. These are the scripts we are going to execute for the preprocessing. Create a folder called preproc for putting the preprocessed videos:  ```mkdir preproc```
create a folder with your experiment name e.g. calsoup inside preproc : ```mkdir preproc/calsoup```
Execute the script : process_videos_job.py which calls process_videos.py. To do this change your parameters for the slurm call in the _job file by changing the -u flag.

Check out if that job executes well. If it does, you can send 10 jobs using that script.

## Face analysis
For face analysis follow these steps:

open the script : AU_analysis_mp_job.sh and adapt the part for "# For ffmpeg local instalation" with your details if you installed ffmpeg locally, or remove this part it if you installed it with conda.

Now create the output dirs, supposing your experiment name is "calsoup"—change as needed:
```
mkdir mp/calsoup/
Execute the script: sbatch --account <YOUR_ACCOUNT> AU_analysis_mp_job.sh
```

Wait for a while and check the outputs to see if it's working. This should create several versions of your videos inside preproc/calsoup/

## External docs & tutorials

- **prepro** — README & examples: https://github.com/ducksouplab/prepro  
- **STIM** — README & tutorial notebooks/sections: https://github.com/Pablo-Arias/STIM  
- **video_analysis** — README & tutorials: https://github.com/Pablo-Arias/video_analysis  
