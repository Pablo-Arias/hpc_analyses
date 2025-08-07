## Configuration 
## conda create --name open_au python=3.8
## conda activate open_au
## module load apps/nvidia-cuda/11.7.1
## pip install --user pandas matplotlib py-feat
## conda develop /users/pa121h/development/STIM
## conda develop /users/pa121h/development/video_analysis
## Run script : sbatch --account project0028 AU_analysis_job.sh

import os
os.environ['OPENBLAS_NUM_THREADS'] = '1'
from face_analysis_pf import analyse_videos

sources = "preproc/brainstorm/*/new_fps/*/*.mp4"

target_folder = "au_analysis/brainstorm/"

analyse_videos(sources
                        , target_folder
                        , skip_frames = 1
                        , batch_size=60
						, num_workers=1 # If running on CPU, you can just leave this to 1!
						, pin_memory=False
						, n_jobs = 1
						, face_model = "retinaface"
						, landmark_model = "mobilefacenet"
						, au_model = 'xgb'
						, emotion_model = "resmasknet"
						, facepose_model = "img2pose"
						, device = "cpu"                    
                    )

print("-------------------- ")
print("---- finished! ----- ")