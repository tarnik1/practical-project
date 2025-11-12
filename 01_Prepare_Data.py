# SCRIPT 01: DATA PREPARATION
# GOAL: Find all relevant FC matrices and save them in a unified format.

# 1. Import necessary libraries
#    (e.g., os, pandas, numpy, glob)

# 2. Define constants
#    - DATA_DIR = 'path/to/your/Data/'
#    - DATASETS_TO_PROCESS = ['TaoWu', 'PPMI', 'Neurocon']
#    - TARGET_FILE_SUFFIX = 'schaefer100_correlation_matrix'
#    - OUTPUT_DIR = 'path/to/processed_data/'

# 3. Initialize an empty list to store metadata
#    - metadata_list = []

# 4. Loop through each dataset in DATASETS_TO_PROCESS
#    - (e.g., for dataset_name in DATASETS_TO_PROCESS:)
#    - Get the path to that dataset's folder.

# 5. Loop through all subject folders in that dataset
#    - (e.g., using os.listdir or glob.glob)
#    - Get the subject_id from the folder name.
#    - Get the diagnosis (PD or control) from the folder/file name.

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