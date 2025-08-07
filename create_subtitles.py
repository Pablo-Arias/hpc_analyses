
## sbatch --account none create_subtitles_job.sh

from transcribe import generate_subtitles


transcription_folder = "transcribed/prolific/*.txt"
target_folder = "subtitles/prolific/"

generate_subtitles(target_folder= target_folder, transcription_folder=transcription_folder)

print("All done for subtitles")