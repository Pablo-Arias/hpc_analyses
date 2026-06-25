# Check the singularity container. 
# You will need three repos : STIM, video_analysis and pyVHR, which can be cloned online : https://github.com/Pablo-Arias, be carefull for pyVHR repo, we are using a patched version, ask Pablo Arias Sarah for the folder
# Execute this scrip with:
# sbatch --account project0028 hr_job.sh

import sys
sys.path.append("/repos/video_analysis")
sys.path.append("/repos/STIM")
sys.path.append("/repos/pyVHR")

import os
os.environ['MPLCONFIGDIR'] = '/tmp' # For matplotlib

from hr_analysis import analyse_folder

wsize = 6

sources = "/mnt/preproc/prolific/*/trimed/*/*.mp4"
analysis_folder = f"/mnt/hr/prolific-meet-up-2026_wsize_{wsize}_updated_filtering_patches_clustering/"

#GPU methods  - don't forget to change _job parameters (gpu) and #SBATCH --gres=gpu:1
#methods = ["HR_CNN", "MTTS_CAN", "cupy_POS", "cupy_CHROM"] 
#cuda = True

#CPU methods - don't forget to change _job parameters
methods = ["cpu_LGI", "cpu_PBV", "cpu_GREEN", "cpu_OMIT", "cpu_ICA", "cpu_SSR", 'cpu_PCA']
cuda = False

bpm_ests = ["median", "clustering"]
#bpm_ests = ["clustering"]

roi_approachs = ["holistic"]
#roi_approachs = ["patches"]

os.makedirs(analysis_folder, exist_ok=True)

for roi_approach in roi_approachs:
	for bpm_est in bpm_ests:
		for method in methods:
			if method in ["HR_CNN", "MTTS_CAN"]:
				target_folder = analysis_folder + method + "/"
			else:
				target_folder = analysis_folder + method + "_" + roi_approach + "_" + bpm_est+"/"

			#if os.path.isdir(target_folder):
			#	print("Skipping, already analysed : " + target_folder)
			#	continue
			
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
						, minHz=0.65 #39 BPM
						, maxHz=2.5 # 150 BPM
						, remove_result_file_if_crashed=False
						, cuda=cuda
						, patch_size = 30
			            )
			
			print("Analysis of target folder DONE : " + target_folder)

print("Analysis Finished ")			

