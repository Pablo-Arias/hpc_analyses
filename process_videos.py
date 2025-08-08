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

experiment = "calsoup"

#sessions = [x.split("/")[-1] for x in glob.glob("original_data/brainstorm/*")]

sessions  =["mkCaltech_videoChat_Game_24" , "mkCaltech_videoChat_Game_25" 
            , "mkCaltech_videoChat_Game_27","mkCaltech_videoChat_Game_28","mkCaltech_videoChat_Game_29","mkCaltech_videoChat_Game_212"
            ,"mkCaltech_videoChat_Game_213","mkCaltech_videoChat_Game_214","mkCaltech_videoChat_Game_215","mkCaltech_videoChat_Game_216"
            ,"mkCaltech_videoChat_Game_218","mkCaltech_videoChat_Game_219","mkCaltech_videoChat_Game_220","mkCaltech_videoChat_Game_221"
            ,"mkCaltech_videoChat_Game_222","mkCaltech_videoChat_Game_223","mkCaltech_videoChat_Game_225","mkCaltech_videoChat_Game_227","mkCaltech_videoChat_Game_228"
            ,"mkCaltech_videoChat_Game_229","mkCaltech_videoChat_Game_230","mkCaltech_videoChat_Game_233","mkCaltech_videoChat_Game_236","mkCaltech_videoChat_Game_237","mkCaltech_videoChat_Game_238"
            ,"mkCaltech_videoChat_Game_239","mkCaltech_videoChat_Game_240","mkCaltech_videoChat_Game_241","mkCaltech_videoChat_Game_242","mkCaltech_videoChat_Game_245"
            ]


for session_name in sessions:
    folder = Path("preproc/"+experiment+ "/"+session_name+"/")
    if folder.is_dir():
        print("Skipping, folder exists")
    else:
        print("Starting : " + session_name)
        ds_process_parallel(sources = "original_data/"+experiment+"/"+session_name+"/*/recordings/", target_folder="preproc/"+experiment+ "/"+session_name+"/")

print("Finished analysis")


