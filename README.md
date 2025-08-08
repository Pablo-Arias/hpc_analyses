# hpc_analyses — Miniforge Environments + Script-by-Script Mapping + SLURM Jobs

BEWARE : the folloiwng read me was written by AI—with some supervision— so may not be completely accurate. Refer to the top of each .py file to make sure you have configured environments correctly. Also, put your data inside original_data/your_experiment. Also, configure each .py file to analyse your data by changing the experiment name to your_experiment.

This repository contains **small, composable pipelines** to preprocess and analyse **DuckSoup** experiment data on laptops and **SLURM** clusters. The goal is to make routine operations (video prep, facial AUs, voice features, transcription→subtitles, heart-rate) **repeatable and scalable**.

- Each script does **one job** (preprocess, AU extraction, audio prep, voice analysis, transcription, subtitles, HR).
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

# 3) Environments overview (create once)

Below are the **core envs** used across the scripts, plus **Face (GPU)** and **Whisper** envs as requested.

> After creating an env, use `conda develop` so `stim`/`prepro` import cleanly (see section 4).

## A) `ds_prepro` —stands for duscksoup preprocessing — video prep, MediaPipe AUs, basic audio

```bash
conda create -y -n ds_prepro python=3.9 -c conda-forge
conda activate ds_prepro
pip install pandas soundfile opencv-python mediapipe tqdm
conda install -y -c conda-forge ffmpeg conda-build
```

## B) `stim39` — advanced audio/voice analysis (via STIM)

```bash
conda create -y -n stim39 python=3.9 -c conda-forge
conda activate stim39
pip install numpy scipy pandas matplotlib soundfile pyloudnorm             opencv-python praat-parselmouth pyo
# optional (some STIM features):
conda install -y -c roebel easdif || true
conda install -y -c conda-forge ffmpeg conda-build
```

## C) `ds_prepro-gpu` — Face/AU GPU variant *(if your AU scripts/jobs use GPU)*

> Only create if your `AU_analysis_GPU.py` + `*_GPU_job.sh` actually need GPU.
> Install the **matching** PyTorch build for your cluster’s CUDA. See https://pytorch.org/get-started/locally/

```bash
conda create -y -n ds_prepro-gpu python=3.9 -c conda-forge
conda activate ds_prepro-gpu
pip install opencv-python mediapipe tqdm
# Install PyTorch matching your CUDA or CPU-only:
# GPU example (change version/index-url to match your cluster):
#   pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
# CPU example:
#   pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
conda install -y -c conda-forge ffmpeg conda-build
```

## D) `whisper` — OpenAI Whisper ASR *(PyTorch backend)*

> Use if your `transcription.py`/job files indicate **OpenAI Whisper**.

```bash
conda create -y -n whisper python=3.10 -c conda-forge
conda activate whisper
# Install PyTorch for your CUDA or CPU as above, then:
pip install openai-whisper
conda install -y -c conda-forge ffmpeg conda-build
```

## E) `fasterwhisper` — Faster‑Whisper ASR *(Ctrnaslate2 backend)*

> Use if your `transcription.py`/job files indicate **Faster‑Whisper**.

```bash
conda create -y -n fasterwhisper python=3.10 -c conda-forge
conda activate fasterwhisper
pip install faster-whisper
conda install -y -c conda-forge ffmpeg conda-build
```

---

# 4) Make `STIM` & `prepro` importable (per env)

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

> If you prefer not to use `conda develop`, add this snippet at the top of your scripts:
>
> ```python
> from pathlib import Path; import sys
> root = Path(__file__).resolve().parent
> sys.path += [str(root.parent / "STIM"), str(root.parent / "prepro")]
> ```

---

# 5) Per‑script environments (from headers + job wrappers)

> **Source of truth:** the **top of each `.py`** and the **associated `*_job.sh`**.  
> This section mirrors those patterns and adds missing Face/Whisper envs. If a script differs in your repo, keep **your** header/job as final.

### `process_videos.py` — Standardize raw recordings
- **Env:** `ds_prepro`
- **Header (env)** *(from script)*
  ```bash
  conda activate ds_prepro
  pip install soundfile pandas opencv-python tqdm
  conda develop "$(pwd)/../STIM"
  conda develop "$(pwd)/../prepro"
  ```
- **Job:** `process_videos_job.sh` → activates `ds_prepro` then runs `python process_videos.py`.

---

### `AU_analysis_mp.py` — Face AUs with MediaPipe (CPU)
- **Env:** `ds_prepro`
- **Header (env)** *(from script)*
  ```bash
  conda activate ds_prepro
  pip install mediapipe opencv-python pandas tqdm
  conda develop "$(pwd)/../STIM"
  conda develop "$(pwd)/../prepro"
  ```
- **Job:** `AU_analysis_mp_job.sh` → activates `ds_prepro` then runs `python AU_analysis_mp.py`.

---

### `AU_analysis.py` / `AU_analysis_GPU.py` — AU alternative / GPU
- **Env:** `ds_prepro-gpu` *(if GPU)*, else `ds_prepro`
- **Header (env)** *(from script; GPU example)*
  ```bash
  conda activate ds_prepro-gpu
  # plus torch/torchvision/torchaudio matching your CUDA (see env C)
  conda develop "$(pwd)/../STIM"
  conda develop "$(pwd)/../prepro"
  ```
- **Jobs:** `AU_analysis_job.sh` / `AU_analysis_GPU_job.sh` → activate the corresponding env and run the script.

---

### `create_au_videos.py` — Render AU overlays on video
- **Env:** `ds_prepro`
- **Header (env)**
  ```bash
  conda activate ds_prepro
  pip install opencv-python pandas tqdm
  conda develop "$(pwd)/../STIM"
  conda develop "$(pwd)/../prepro"
  ```
- **Job:** `create_au_videos_job.sh`

---

### `process_audio.py` — Audio extraction/normalization
- **Env:** `ds_prepro`
- **Header (env)**
  ```bash
  conda activate ds_prepro
  pip install soundfile pandas tqdm
  conda develop "$(pwd)/../STIM"
  conda develop "$(pwd)/../prepro"
  ```
- **Job:** `process_audio_job.sh`

---

### `voice_analysis.py` — Voice/acoustic features (via STIM)
- **Env:** `stim39`
- **Header (env)**
  ```bash
  conda activate stim39
  pip install numpy scipy pandas matplotlib soundfile pyloudnorm               opencv-python praat-parselmouth pyo
  conda develop "$(pwd)/../STIM"
  conda develop "$(pwd)/../prepro"
  ```
- **Job:** `voice_analysis_job.sh`

---

### `transcription.py` — Automatic speech recognition (Whisper backends)
- **Env (choose based on your script/job):**
  - **OpenAI Whisper:** `whisper` (section D)
  - **Faster‑Whisper:** `fasterwhisper` (section E)
- **Header (env)** *(example for Faster‑Whisper)*
  ```bash
  conda activate fasterwhisper
  pip install faster-whisper
  conda develop "$(pwd)/../STIM"
  conda develop "$(pwd)/../prepro"
  ```
  *(or for OpenAI Whisper use `conda activate whisper` and `pip install openai-whisper`)*
- **Job:** `transcription_job.sh` (or `transcribe_test.py` for quick checks)

---

### `create_subtitles.py` — Generate subtitles (VTT/SRT)
- **Env:** `ds_prepro`
- **Header (env)**
  ```bash
  conda activate ds_prepro
  pip install pandas tqdm
  conda develop "$(pwd)/../STIM"
  conda develop "$(pwd)/../prepro"
  ```
- **Job:** `create_subtitles_job.sh`

---

### `hr.py` — Heart‑rate extraction
- **Env:** `ds_prepro` (+ Apptainer/Singularity if a `.sif` is required)
- **Header (env)**
  ```bash
  conda activate ds_prepro
  pip install pandas tqdm
  conda develop "$(pwd)/../STIM"
  conda develop "$(pwd)/../prepro"
  # if the header mentions a .sif: place it under models/ (e.g., models/interalysis0.1.sif)
  ```
- **Job:** `hr_job.sh`

---

### `index.py` — Index/QA of processed outputs
- **Env:** `ds_prepro`
- **Header (env)**
  ```bash
  conda activate ds_prepro
  pip install pandas tqdm
  conda develop "$(pwd)/../STIM"
  conda develop "$(pwd)/../prepro"
  ```
- **Job:** `index_job.sh`

---

# 6) Data & models layout

- **Raw recordings:** `original_data/<EXPERIMENT_NAME>/…`
- **Large models/containers:** `models/` (e.g., `.sif` for HR if used)
- **Outputs:** each script writes to analysis‑specific folders (see header defaults)

Bootstrap your tree:
```bash
mkdir -p original_data/<EXPERIMENT_NAME>/
mkdir -p models/ logs/
```

---

# 7) SLURM: use the job scripts already in this repo

Use the provided `_job.sh` wrappers **as‑is**. Typically you only edit:
- `#SBATCH --account=…` (your project)
- `#SBATCH --partition=…` (queue/partition)
- time/mem/CPU (and `--gres=gpu:…` if GPU)
- the **Miniforge activation** (two options below)

Put one of these near the top **before** `conda activate`:

```bash
# Option A (recommended)
source "$HOME/miniforge3/etc/profile.d/conda.sh"
# Option B (portable)
eval "$($HOME/miniforge3/bin/conda shell.bash hook)"
```

**Submit & monitor**
```bash
mkdir -p logs
sbatch --account <YOUR_ACCOUNT> process_videos_job.sh
squeue -u "$USER"
tail -f logs/<jobname>-<jobid>.out
seff <JOBID>
sstat -j <JOBID> --format=JobID,MaxRSS,AveRSS,MaxVMSize
```

## External docs & tutorials

- **prepro** — README & examples: https://github.com/ducksouplab/prepro  
- **STIM** — README & tutorial notebooks/sections: https://github.com/Pablo-Arias/STIM  
- **video_analysis** — README & tutorials: https://github.com/Pablo-Arias/video_analysis  
