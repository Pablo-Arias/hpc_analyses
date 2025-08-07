# env : conda activate stim39
#execute  : sbatch --account project0028 process_audio_job.sh

from ducksoup import ds_process_audio_only

import glob

for source_folder in glob.glob("original_data/ultimatum/mkUltimatum_Game_Real1/*/recordings/"):
    print(source_folder)
    folder_tag = source_folder.split("/")[-3] + "/"   
    ds_process_audio_only(source_folder=source_folder
                            , folder_tag=folder_tag
                            , target_folder="audio_preproc/ultimatum/"
                            )