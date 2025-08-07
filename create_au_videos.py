## Check process_videos.py to setup environment
## conda create --name feat-au python=3.9 
## conda activate feat-au
## pip install py-feat opencv-python pandas matplotlib
## conda develop /users/pa121h/development/STIM
## conda develop /users/pa121h/development/video_analysis
## Sometimes, I have problems with pip, I have to conda uninstall pip, pip uninstall pip, then conda install pip, then de/reactivate enbironment for pip to work
## Execute script here : sbatch --account none create_au_videos_job.sh

from face_analysis import create_au_video, create_tracked_video
from video_processing import combine_2_videos
from conversions import get_file_without_path
import os
import glob


data_id = "schyns"
combined_folder= "preproc/"+data_id+"/combined_au_anlaysis/"

for au_analysis in glob.glob("au_analysis/"+data_id+"/*.csv"):
    if os.stat(au_analysis).st_size == 0:
        continue

    #get_file tag
    file_tag = get_file_without_path(au_analysis)
    print("Starting generation of : " + file_tag)

    # Create tracked videos
    print("Start tracking")
    target_tracked_video_folder  = "preproc/"+data_id+"/tracked/"
    target_frames_folder = "preproc/schyns/frames/tracked/"+file_tag + "/"
    create_tracked_video(au_analysis, target_video_folder=target_tracked_video_folder, target_frames_folder=target_frames_folder, extract_frames=True, fps=30, remove_frames=True, add_audio=True)

    #Create AU videos
    print("Start create AU videos")
    target_frames_folder = "preproc/"+data_id+"/frames/au/" + file_tag +"/"
    target_au_video_folder = "preproc/"+data_id+"/video_au/"
    create_au_video(au_analysis, target_video_folder=target_au_video_folder, target_frames_folder=target_frames_folder, extract_frames=True, fps=30, remove_frames=True, add_audio=True)

    #Combine videos
    print("Start combine videos")
    tracked_video   = target_tracked_video_folder + file_tag + ".mp4"
    au_video        = target_au_video_folder + file_tag + ".mp4"
    os.makedirs(combined_folder, exist_ok=True)
    combine_2_videos(tracked_video, au_video, combined_folder + file_tag + ".mp4", combine_audio_flag=True)
