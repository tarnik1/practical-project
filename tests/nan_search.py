import os
import numpy as np

DATA_PATH = r'C:\Users\nikna\Documents\pp_datasets\processed_data_1' # Change this to your folder
files = [f for f in os.listdir(DATA_PATH) if f.endswith('.npy')]

problematic_files = []

print(f"Scanning {len(files)} files for NaNs...")

for f in files:
    file_path = os.path.join(DATA_PATH, f)
    matrix = np.load(file_path)
    
    if np.isnan(matrix).any() or np.isinf(matrix).any():
        count = np.isnan(matrix).sum() + np.isinf(matrix).sum()
        print(f" [!] ISSUE FOUND: {f} ({count} bad values)")
        problematic_files.append(f)

if not problematic_files:
    print("Success: No NaNs or Infs found in any .npy files!")
else:
    print(f"\nScan complete. Found {len(problematic_files)} problematic files.")

    