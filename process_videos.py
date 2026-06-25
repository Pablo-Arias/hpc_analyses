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
from pathlib import Path


print("Starting analysis")

experiment = "noah"

#sessions = [x.split("/")[-1] for x in glob.glob("original_data/brainstorm/*")]

sessions  =["mknoah_brainstorming_session1",   "mknoah_brainstorming_session2",  "mknoah_brainstorming_session5",  "mknoah_brainstorming_session9",    "mknoah_brainstorming_session_23","mknoah_brainstorming_session10",  "mknoah_brainstorming_session3",  "mknoah_brainstorming_session6",  "mknoah_brainstorming_session_21","mknoah_brainstorming_session11",  "mknoah_brainstorming_session4",  "mknoah_brainstorming_session7",  "mknoah_brainstorming_session_22"]

for session_name in sessions:
    folder = Path("preproc/"+experiment+ "/"+session_name+"/")
    if folder.is_dir():
        print("Skipping, folder exists")
    else:
        print("Starting : " + session_name)
        ds_process_parallel(sources = "original_data/"+experiment+"/"+session_name+"/*/recordings/", target_folder="preproc/"+experiment+ "/"+session_name+"/")

print("Finished analysis")


