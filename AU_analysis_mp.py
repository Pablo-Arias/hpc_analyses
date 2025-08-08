## Configuration 
## conda create --name mediapipe python=3.11
## conda activate mediapipe
## pip install mediapipe scipy soundfile polars
## conda develop /mnt/data/project0028/repos/STIM
## conda develop /mnt/data/project0028/repos/video_analysis
## Run script : sbatch --account project0028 AU_analysis_mp_job.sh

import sys
sys.path.insert(0,'/mnt/data/project0028/repos/STIM')
sys.path.insert(0,'/mnt/data/project0028/repos/video_analysis')

from face_analysis_mp import analyse_video_parallel, analyse_video
import glob
import os
import time

model = "models/face_landmarker_v2_with_blendshapes.task"

# detection_results = analyse_video_parallel(sources= "preproc/prolific/*/trimed/*/*.mp4"
#                                   , target_analysis_folder   = "mp/au_analysis/"
#                                   , target_frames_folder     = "mp/tracked/"
#                                   , target_video_folder      = "mp/tracked_video/"
#                                   , target_au_video_folder   = "mp/au_video/"
#                                   , target_AU_plots_folder   = "mp/AU_bar_graph_folder/"
#                                   , combined_videos_folder   = "mp/combined_videos_folder/"
#                                   , target_processing_folder = "mp/processing/" 
#                                   , model_asset_path         = model
#                                   , export_tracked_frames    = True
#                                   , delete_frames            = True
#                                   , delete_bar_graphs        = True
#                                   , export_blendshapes       = True
#                                   , export_lmks              = False
#                                   , export_AU_bargraphs      = True
#                                   , create_tracked_video     = True
#                                   , combine_AU_graphs_into_video = True
#                                   , combine_AU_bargraphs_and_tracked_video = True
#                                   )


#One file at a time
sources = "preproc/calsoup/*/trimed/*/*.mp4"

#sources = "original_data/speed_dating/*.mp4"

start_total = time.perf_counter()                       # timer for the whole batch
for file in glob.glob(sources):
    print("Processing file : " + file)
    
    t0 = time.perf_counter()                            # timer for this file
    analyse_video(file
                        , target_analysis_folder   = "mp/calsoup/au_analysis/"
                        , target_frames_folder     = "mp/calsoup/tracked/"
                        , target_video_folder      = "mp/calsoup/tracked_video/"
                        , target_au_video_folder   = "mp/calsoup/au_video/"
                        , target_AU_plots_folder   = "mp/calsoup/AU_bar_graph_folder/"
                        , combined_videos_folder   = "mp/calsoup/combined_videos_folder/"
                        , target_processing_folder = "mp/calsoup/processing/" 
                        , model_asset_path         = model
                        , export_tracked_frames    = False
                        , delete_frames            = False
                        , delete_bar_graphs        = False
                        , export_blendshapes       = True
                        , export_lmks              = False
                        , export_AU_bargraphs      = False
                        , create_tracked_video     = False
                        , combine_AU_graphs_into_video = False
                        , combine_AU_bargraphs_and_tracked_video = False
                        )
    elapsed = time.perf_counter() - t0
    print(f"  ↳ finished {os.path.basename(file)} in {elapsed:,.1f} s")

    

print("Finished!!!")