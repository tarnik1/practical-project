import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os

# Define the output directory from your data preparation script
output_dir = r'C:\Users\nikna\Documents\pp_datasets\processed_data_1'
master_csv_path = os.path.join(output_dir, "master_metadata.csv")

# --- NEW: Create a directory for plots within the output directory ---
plots_output_dir = os.path.join(output_dir, 'Plots')
os.makedirs(plots_output_dir, exist_ok=True) # Ensure the directory exists

# --- Load the Master Metadata ---
try:
    master_df = pd.read_csv(master_csv_path)
    print(f"Loaded master metadata with {len(master_df)} subjects.")
except FileNotFoundError:
    print(f"Error: Could not find the master metadata file at {master_csv_path}.")
    exit()

# --- Select One Subject (Same as before) ---
# CHANGE INDEX HERE (e.g., master_df.iloc[5]) to select a different subject
subject_data = master_df.iloc[100] 
subject_id = subject_data['subject_id']
matrix_path = subject_data['npy_path']
diagnosis = subject_data['diagnosis']

print(f"\nplotting FC matrix for:")
print(f"  Subject ID: {subject_id}")
print(f"  Diagnosis:  {diagnosis}")

# --- Load the Matrix ---
try:
    fc_matrix = np.load(matrix_path)
    if fc_matrix.shape != (100, 100):
        print(f"Warning: Expected a 100x100 matrix, but loaded one with shape {fc_matrix.shape}.")
except FileNotFoundError:
    print(f"Error: Could not find the matrix file at {matrix_path}.")
    exit()
except Exception as e:
    print(f"Error loading the matrix: {e}")
    exit()

# --- Step 4: Generate the Heatmap Plot ---
plt.figure(figsize=(10, 8)) # Set figure size

# Core plotting function
heatmap = plt.imshow(fc_matrix, cmap='viridis', vmin=-1, vmax=1)

# Colorbar for correlation coefficient legend
plt.colorbar(heatmap, label='Correlation Coefficient')

# Adjust layout and cleaner ticks (same as before)
plt.xticks(np.arange(0, 101, 20)) 
plt.yticks(np.arange(0, 101, 20))
plt.tight_layout() 

# --- NEW: Define a filename and path for the saved plot ---
# Incorporate subject ID and diagnosis into the filename
plot_filename = f"{subject_id}_{diagnosis.replace(' ', '_')}_fc_matrix.png"
plot_save_path = os.path.join(plots_output_dir, plot_filename)

# --- NEW: Save the plot BEFORE showing it ---
# dpi=300 sets high resolution, bbox_inches='tight' crops whitespaces
plt.savefig(plot_save_path, dpi=300, bbox_inches='tight')
print(f"\nPlot saved successfully to: {plot_save_path}")

# Display the plot in the pop-up window as well
plt.show() 

print("\nPlot displayed and saved.")