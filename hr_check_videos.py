import subprocess
import glob
import os

def check_video_framerate_stability(video_path):
    """
    Measures the duration of every frame in the video.
    Returns True if CFR, False if VFR.
    """
    cmd = [
        "ffprobe", "-v", "error", "-select_streams", "v:0",
        "-show_entries", "frame=pkt_duration_time",
        "-of", "default=noprint_wrappers=1:nokey=1",
        video_path
    ]
    
    try:
        # Extract all frame durations
        result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        if result.returncode != 0:
            return "Read Error"

        durations = result.stdout.strip().split('\n')
        unique_durations = set([d for d in durations if d])
        
        # A perfectly CFR video will have 1 unique duration. 
        # We allow up to 3 to account for minor container metadata rounding.
        if len(unique_durations) <= 3:
            return f"CFR PASS (Unique durations: {len(unique_durations)})"
        else:
            return f"VFR FAIL (Unique durations: {len(unique_durations)})"
            
    except Exception as e:
        return f"Execution Error: {e}"

if __name__ == "__main__":
    # Point this to your newly re-encoded directory
    search_path = "//mnt/data/project0028/video_analysis/preproc/prolific/*/trimed/*/*.mp4"
    videos = glob.glob(search_path)
    
    print(f"Found {len(videos)} videos to verify.\n")
    
    failed_videos = []
    
    for vid in videos:
        status = check_video_framerate_stability(vid)
        print(f"{status} | {os.path.basename(vid)}")
        
        if "FAIL" in status or "Error" in status:
            failed_videos.append(vid)
            
    print("\n--- Summary ---")
    print(f"Total passed: {len(videos) - len(failed_videos)}")
    print(f"Total failed/VFR: {len(failed_videos)}")