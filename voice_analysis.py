## create environment:
## In a safe folder : git clone https://github.com/Pablo-Arias/STIM.git
## module load apps/anaconda3
## module load apps/ffmpeg/6.0.0/gcc-4.8.5+x264
## conda create --name stim39 python=3.9
## conda activate stim39
## conda install -c roebel easdif
## conda develop /mnt/data/project0028/repos/STIM
## conda develop /mnt/data/project0028/repos/video_analysis
## conda develop /mnt/data/project0028/repos/prepro
## pip install scipy soundfile pyloudnorm pandas pyo praat-parselmouth matplotlib numpy opencv-python
## If you want super-vp dependencies, you need a super-vp licence.
## Get it here : https://forum.ircam.fr/projects/detail/analysissynthesis-command-line-tools/
## And authorisation file in your account
## Execute script here : sbatch --account project0028 voice_analysis_job.sh

from video_processing import extract_audio_folder
from audio_analysis import analyse_audio_ts_folder, analyse_audio_folder_parallel
import pandas as pd


def analyse_audio():
	#First extract audio from videos and convert to mono
	
	data_id = "calsoup"
	source     = "preproc/"+data_id+"/*/trimed/*/*.mp4"

	extract_audio_folder(source, target_folder= "extracted_audio/"+data_id+"/", nb_audio_channels=1)

	#Extract audoio time series, this works in parallel in sevral CPU cores
	analyse_audio_ts_folder(source_folder= "extracted_audio/"+data_id+"/*.wav"
	 							, time_step				= 0.01
	 							, praat_ws			    = 0.04
	 							, sc_ws                 = 1024
	 							, rms_ws                = 1024
	 							, pitch_floor			= 75
	 							, pitch_ceiling			= 450
	 							, nb_formants			= 5
	 							, max_formant_freq		= 5500
	 							, pre_emph				= 50.0
	 							, parameter_tag         = None
	 							, verbose               = True
	 							, silence_threshold      = -50
	 							, target_folder			 = "audio_analysis_ts/"+data_id+"/"
	 							, plot_features          = True
	 				)

	print("time series analysis finished")

	#Extract audo time series
	#source_folder = "extracted_audio/"+data_id+"/*.wav"
	# analyse_audio_folder_parallel(source_folder
	# 						, speed_of_sound		= 335
	# 						, time_step				= 0.01
	# 						, window_size			= 0.01 
	# 						, pitch_floor			= 75
	# 						, pitch_ceiling			= 350
	# 						, nb_formants			= 5
	# 						, nb_formants_fd		= 5
	# 						, max_formant_freq		= 5500
	# 						, pre_emph				= 50.0
	# 						, harmonicity_threshold = 0.3 
	# 						, sc_rms_thresh         = -40
	# 						, sc_ws                 = 512
	# 						, parameter_tag         = None
	# 						, verbose               = True
	# 						, formant_method		 = 'median'
	# 						, target_folder			 = "audio_analysis/"+data_id +"/"
	# 						)

	print("All done for audio analysis!")
							

if __name__ == "__main__":
	print("starting analysis")
	analyse_audio()