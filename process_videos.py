# env : conda activate stim39
# Import ffmpeg local instalation
#export INSTALL_DIR="/mnt/data/project0028"
#export SRC_DIR="$INSTALL_DIR/ffmpeg_sources"
#export PATH="$INSTALL_DIR/bin:$PATH"
#export PKG_CONFIG_PATH="$INSTALL_DIR/lib/pkgconfig"

#execute  : sbatch --account project0028 process_videos_job.sh

## -- Process all videos
from ducksoup import ds_process_parallel
import glob

print("Starting analysis")

experiment = "brainstorm"

#sessions = [x.split("/")[-1] for x in glob.glob("original_data/brainstorm/*")]

sessions  =["mkchristos_brainstorming_session7"]

for session_name in sessions:
    ds_process_parallel(sources = "original_data/"+experiment+"/"+session_name+"/*/recordings/", target_folder="preproc/"+experiment+ "/"+session_name+"/")

print("Finished analysis")


