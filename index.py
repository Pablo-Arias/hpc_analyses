## Documentation here : https://github.com/openai/whisper
## Models : https://github.com/openai/whisper#available-models-and-languages

## conda activate whisper
## sbatch --account project0028 index_job.sh

from video_processing import index_video_recordings_parallel

sources = "preproc/re-encode/*/*.mp4"
indexed_path = "indexed_audio/"
index_video_recordings_parallel(sources, rms_threshold = -50, WndSize = 16384, indexed_path = indexed_path, add_time_tag=True)
