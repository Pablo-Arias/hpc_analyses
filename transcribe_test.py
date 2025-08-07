## Configuration 
## module load apps/anaconda3
##module load apps/ffmpeg/6.0.0/gcc-4.8.5
## conda activate whisper
## python transcribe_test.py

from transcribe import transcribe_video_file_time_stamps

file = "preproc/prolific/mkreal_meeting_experiment_prolific3/trimed/1-p2p6/i-be39996e91202f30a779e1e17edd3773-a-20240202-153811.004-s-mkreal_meeting_experiment_prolific3-n-1-p2p6-u-p6-c-1-dry.mp4"

transcribe_video_file_time_stamps(file, transcription_path="test_transcribed/"
                                      , audio_path= "extracted_audio/"
                                      , model_type= "large-v2"
                                      , language="En"
                                      , device="cpu"
                                      , extract_audio_flag=True
                                      , temperature = (0.0, 0.2, 0.4, 0.6, 0.8, 1.0)
                                      , best_of=5
                                      , beam_size=5
                                      , vad="auditok"
                                      , detect_disfluencies=True
                                      )