# SCRIPT 01: DATA PREPARATION
# GOAL: Find all relevant FC matrices and save them in a unified format.

# 1. Import necessary libraries
#    (e.g., os, pandas, numpy, glob)
import os
import pandas as pd
import numpy as np
import glob
# 2. Define constants
#    - DATA_DIR = 'path/to/your/Data/'
#    - DATASETS_TO_PROCESS = ['TaoWu', 'PPMI', 'Neurocon']
#    - TARGET_FILE_SUFFIX = 'schaefer100_correlation_matrix'
#    - OUTPUT_DIR = 'path/to/processed_data/'
metadata_path = 'C:\Users\nikna\Documents\pp_datasets\Metadata_v5-20251021T115759Z-1-001\Metadata_v5\Metadata_v5'
neurocon_path = 'C:\Users\nikna\Documents\pp_datasets\neurocon-20251021T115804Z-1-001\neurocon\neurocon'
ppmi_path = 'C:\Users\nikna\Documents\pp_datasets\ppmi_v2-20251021T115808Z-1-001\ppmi_v2\ppmi'
taowu_path = 'C:\Users\nikna\Documents\pp_datasets\taowu-20251021T115812Z-1-001\taowu\taowu'

metadata_files = ["Neurocon_metadata.csv", "PPMI_metadata.csv", "TaoWu_metadata.csv"]
targetfile_suffix = 'schaefer100_correlation_matrix'
output_dir = ''
# 3. Initialize an empty list to store metadata
#    - metadata_list = []
metadata_list = []
#mapping metadata files to the path for the dataset folders
dataset_map = {
    "Neurocon_metadata.csv": neurocon_path,
    "PPMI_metadata.csv":     ppmi_path,
    "TaoWu_metadata.csv":    taowu_path,
}

# 4. Loop through each dataset in DATASETS_TO_PROCESS
#    - (e.g., for dataset_name in DATASETS_TO_PROCESS:)
#    - Get the path to that dataset's folder.
for metadata_file in metadata_files:
    print("processing metadata file:", metadata_file)
    metadata_file_path = os.path.join(metadata_path, metadata_file)
    if not os.path.exists(metadata_file_path):
        print("metadata file not found:", metadata_file_path)
        continue

    #loading CSV
    meta_df = pd.read_csv(metadata_file_path)
    id_col = "Subject"
    if id_col not in meta_df.columns:
        print(f"no 'Subject' column found in:", metadata_file_path)
        continue
    #getting the dataset path for this metadata file
    dataset_path = dataset_map.get(metadata_file)
    print("using dataset folder:", dataset_path)
# 5. Loop through all subject folders in that dataset
#    - (e.g., using os.listdir or glob.glob)
#    - Get the subject_id from the folder name.
#    - Get the diagnosis (PD or control) from the folder/file name.
     subject_ids = meta_df[id_col].astype(str).tolist()
     
# 6. Find the target file
#    - Search inside the subject's folder for the file that ends with
#      TARGET_FILE_SUFFIX (and is a correlation_matrix, not timeseries).
#    - (e.g., sub-control032057_schaefer100_correlation_matrix)

# 7. If the target file is found:
#    - a. Load the matrix (e.g., using numpy.loadtxt(file_path)).
#    - b. Define a new, clean save path for this subject's data
#         (e.g., OUTPUT_DIR / f"{subject_id}_fc_matrix.npy").
#    - c. Save the loaded matrix as a .npy file (numpy.save).
#    - d. Append this subject's info to metadata_list:
#         {
#           'subject_id': subject_id,
#           'diagnosis': 'PD' or 'Control',
#           'dataset_source': dataset_name,
#           'npy_path': new_save_path
#         }

# 8. After all loops are finished:
#    - Convert metadata_list to a pandas DataFrame.
#    - Save this DataFrame to a CSV file (e.g., OUTPUT_DIR / 'metadata.csv').
#    - This CSV will be the "master file" for all other scripts.

# 9. Print a success message
#    - (e.g., "Data preparation complete. Processed X subjects.")