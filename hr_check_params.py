import sys
sys.path.append("/repos/video_analysis")
sys.path.append("/repos/STIM")
sys.path.append("/repos/pyVHR")

import inspect
from pyVHR.analysis.pipeline import Pipeline

def check_pyvhr_params():
    print("Inspecting Pipeline.run_on_video...")
    
    # Extract the function signature
    sig = inspect.signature(Pipeline.run_on_video)
    
    # Check for specific arguments
    has_min = 'minHz' in sig.parameters
    has_max = 'maxHz' in sig.parameters
    
    if has_min and has_max:
        print("\n✅ SUCCESS: Your pyVHR version accepts 'minHz' and 'maxHz' directly.")
    else:
        print("\n❌ WARNING: 'minHz' and 'maxHz' are NOT direct parameters in your version.")
        
    print("\n--- Available Parameters ---")
    for name, param in sig.parameters.items():
        print(f"  - {name}")

if __name__ == "__main__":
    check_pyvhr_params()