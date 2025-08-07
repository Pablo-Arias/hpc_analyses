# Check the singularity container. 
# You will need three repos : STIM, video_analysis and pyVHR
# Execute this scrip with:
# sbatch --account project0028 hr_job.sh

import sys
sys.path.append("/repos/video_analysis")
sys.path.append("/repos/STIM")
sys.path.append("/repos/pyVHR")

import os
os.environ['MPLCONFIGDIR'] = '/tmp' # For matplotlib

from hr_analysis import analyse_folder

sources = "/mnt/preproc/prolific/*/new_fps/*/*.mp4"
analysis_folder = "/mnt/hr/prolific_meet_up/"

methods = ["HR_CNN", "MTTS_CAN", "cupy_POS", "cupy_CHROM", "cpu_LGI", "cpu_PBV", "cpu_GREEN", "cpu_OMIT", "cpu_ICA", "cpu_SSR", 'cpu_PCA', "MTTS_CAN", "HR_CNN"] 

bpm_ests = ["median", "clustering"]
roi_approachs = ["holistic"]

wsize = 8

os.makedirs(analysis_folder, exist_ok=True)

for roi_approach in roi_approachs:
	for bpm_est in bpm_ests:
		for method in methods:
			if method in ["HR_CNN", "MTTS_CAN"]:
				target_folder = analysis_folder + method + "/"
			else:
				target_folder = analysis_folder + method + "_" + roi_approach + "_" + bpm_est+"/"
			
			if os.path.isdir(target_folder):
				print("Skipping, already analysed : " + target_folder)
				continue
			
			#Create analysis folder
			print("Starting analysis " + target_folder)
			os.makedirs(target_folder, exist_ok=True)
			
			#analyse folder
			analyse_folder(sources
						, target_folder
						, bpm_est=bpm_est
						, method = method
						, roi_approach = roi_approach
						, wsize = wsize
			            )

