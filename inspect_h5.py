import h5py
import numpy as np

# Point this to your actual file path
file_path = "data/libero_spatial/episode_0000_20260418_201843_193953.h5"

print(f"--- Opening {file_path} ---")

with h5py.File(file_path, "r") as f:
    # 1. Read the high-level Episode Metadata
    print("\n[EPISODE ATTRIBUTES]")
    for key, value in f.attrs.items():
        print(f"  {key}: {value}")

    # 2. Look at how many steps were recorded
    steps = list(f.keys())
    print(f"\n[RECORDED STEPS]: {len(steps)} total steps found.")

    # 3. Crack open the very first step to see the actual arrays
    if len(steps) > 0:
        first_step = steps[0]
        print(f"\n[CONTENTS OF {first_step}]")
        
        step_group = f[first_step]
        for key in step_group.keys():
            data = step_group[key]
            
            # Print the shape and data type of the array
            print(f"  - {key}: shape={data.shape}, dtype={data.dtype}")
            
            # If it's the physical state or action, print the actual numbers!
            if key in ["robot_state", "clean_action"]:
                print(f"      Values: {np.array(data)}")
            
            # If it's the dummy embeddings we injected, verify they are zeros
            if key in ["prompt_emb", "vision_0"]:
                preview = np.array(data)[0, :4] # Just look at the first 4 numbers
                print(f"      Preview: {preview} ... (Expected: [0. 0. 0. 0.])")
