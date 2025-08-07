## Configuration 
## module load apps/anaconda3
## conda create --name open_au_gpu python=3.8 
## conda activate open_au_GPU
## module load apps/nvidia-cuda/12.0.1
## conda develop /users/pa121h/development/STIM
## conda develop /users/pa121h/development/video_analysis
## pip install --user pandas matplotlib py-feat torch==2.0.1 
## pip3 install --user torchvision==0.15.2
## Run script : sbatch --account project0028 AU_analysis_GPU_job.sh

import os
os.environ['OPENBLAS_NUM_THREADS'] = '1'
from face_analysis_pf import analyse_videos

sources = "preproc/brainstorm/*/new_fps/*/*.mp4"

target_folder = "au_analysis/brainstorm/"

analyse_videos(sources
                        , target_folder
                        , skip_frames = 1
                        , batch_size=100
						, num_workers=8 # number of cores
						, pin_memory=False
						, n_jobs = 1
						, face_model = "retinaface"
						, landmark_model = "mobilefacenet"
						, au_model = 'xgb'
						, emotion_model = "resmasknet"
						, facepose_model = "img2pose"
						, device = "cuda"
                    )

print("-------------------- ")
print("---- finished! ----- ")