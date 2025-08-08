## Documentation here : https://github.com/openai/whisper
## Models : https://github.com/openai/whisper#available-models-and-languages

## Configuration 
## conda create --name whisper python=3.11
## conda activate whisper
## pip install whisper
## pip3 install whisper-timestamped
## pip install opencv-python
## pip install auditok
## conda develop /users/pa121h/development/STIM
## conda develop /users/pa121h/development/video_analysis
## sbatch --account project0028 transcription_job.sh

import whisper
from video_processing import extract_audio
from transcribe import transcribe_parallel, transcribe_parallel_time_stamps, transcribe_video_file_time_stamps
import glob
from conversions import get_file_without_path
import glob

model       = "large-v3" #"large" #or use medium
target_path = "transcribed/"
data_id     = "calsoup/"
sources     = "preproc/"+data_id+"/*/trimed/*/*.mp4"

#Process one file at a time
for file in glob.glob(sources):
    transcribe_video_file_time_stamps(file                                    
                                    , transcription_path = target_path + data_id
                                    , audio_path = "extracted_audio/"+ data_id
                                    , model_type= model
                                    , language="English"
                                    , device="cpu"
                                    , extract_audio_flag = False
                                    , temperature = (0.0, 0.2, 0.4, 0.6, 0.8, 1.0)
                                    , best_of   = 5
                                    , beam_size = 5
                                    , vad       = "auditok"
                                    , detect_disfluencies = True
                                    )

#Transcriibe parallel
#Transcribe with time stamps; See parameters here : https://github.com/linto-ai/whisper-timestamped#light-installation-for-cpu
#transcribe_parallel_time_stamps(sources
#                                    , transcription_path = target_path + data_id
#                                    , audio_path = "extracted_audio/"+ data_id
#                                    , model_type= model
#                                    , language="English"
#                                    , device="cpu"
#                                    , extract_audio_flag = True
#                                    , temperature = (0.0, 0.2, 0.4, 0.6, 0.8, 1.0)
#                                    , best_of   = 5
#                                    , beam_size = 5
#                                    , vad       = "auditok"
#                                    , detect_disfluencies = True
#                                    )

# Without time stamps
#transcribe_parallel(sources, model_type=model, audio_path= "extracted_audio/")

print("finished!")