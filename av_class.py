# SCRIPT 02: GENERATE GROUP AVERAGE FC MATRICES
# GOAL: Plot the average FC matrix for PD subjects and Healthy Controls.

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os

# Define the output directory where pre-processed data is saved
# This must match output_dir from your SCRIPT 01
output_dir = r'C:\Users\nikna\Documents\pp_datasets\processed_data_1'
master_csv_path = os.path.join(output_dir, "master_metadata.csv")

# --- Step 1: Load the Master Metadata ---
try:
    master_df = pd.read_csv(master_csv_path)
    print(f"Loaded master metadata with {len(master_df)} subjects.")
except FileNotFoundError:
    print(f"Error: Could not find the master metadata file at {master_csv_path}.")
    print("Please make sure you have run SCRIPT 01: DATA PREPARATION successfully.")
    exit()

# --- Step 2: Initialize Arrays for Summing Matrices ---
# The matrices are 100x100 based on the Schaefer 100 atlas
pd_sum_matrix = np.zeros((100, 100))
healthy_sum_matrix = np.zeros((100, 100))
pd_count = 0
healthy_count = 0

print("\nProcessing subjects and calculating average matrices...")

# --- Step 3: Loop Through Metadata and Accumulate Matrices ---
for index, row in master_df.iterrows():
    label = row['label']
    matrix_path = row['npy_path']
    subject_id = row['subject_id']

    try:
        # Load the subject's .npy FC matrix
        fc_matrix = np.load(matrix_path)

        # Basic shape check to prevent issues
        if fc_matrix.shape != (100, 100):
            print(f"Warning: Subject {subject_id} has a matrix of shape {fc_matrix.shape}. Skipping.")
            continue

        # Add to the correct group sum based on the label
        if label == 1: # PD subject
            pd_sum_matrix += fc_matrix
            pd_count += 1
        elif label == 0: # Healthy Control subject
            healthy_sum_matrix += fc_matrix
            healthy_count += 1
        else:
            print(f"Warning: Subject {subject_id} has an invalid label: {label}. Skipping.")

    except FileNotFoundError:
        print(f"Error: Matrix file not found for subject {subject_id} at path {matrix_path}. Skipping.")
    except Exception as e:
        print(f"Error processing subject {subject_id}: {e}. Skipping.")

print(f"\nProcessing complete:")
print(f"  Found {pd_count} subjects with PD.")
print(f"  Found {healthy_count} healthy control subjects.")

# --- Step 4: Calculate Average Matrices ---
# Be careful not to divide by zero if a group is empty
if pd_count > 0:
    pd_average_matrix = pd_sum_matrix / pd_count
else:
    pd_average_matrix = np.zeros((100, 100))
    print("\nWarning: No subjects found with PD. Average matrix is all zeros.")

if healthy_count > 0:
    healthy_average_matrix = healthy_sum_matrix / healthy_count
else:
    healthy_average_matrix = np.zeros((100, 100))
    print("\nWarning: No healthy control subjects found. Average matrix is all zeros.")

# --- Step 5: Generate the Plots ---
# Create a figure with two subplots side-by-side
fig, axes = plt.subplot_mosaic([['PD', 'Healthy']], figsize=(16, 7))

# Parameters for plotting the matrices as heatmaps
# 'imshow' displays data as an image.
# cmap='viridis' is a perceptually uniform colormap suitable for general data visualization.
# vmin=-1, vmax=1 sets the colorbar limits to capture the full range of correlations.
imshow_kwargs = {'cmap': 'viridis', 'vmin': -1, 'vmax': 1}

# Plot 1: Average PD Matrix
pd_plot = axes['PD'].imshow(pd_average_matrix, **imshow_kwargs)
axes['PD'].set_title(f'Average FC Matrix - Parkinson\'s Disease\n(N={pd_count})')
axes['PD'].set_xlabel('regions')
axes['PD'].set_ylabel('regions')
fig.colorbar(pd_plot, ax=axes['PD'], label='Correlation Coefficient')
# Customize tick locations to cleaner, e.g., every 20 regions
axes['PD'].set_xticks(np.arange(0, 101, 20))
axes['PD'].set_yticks(np.arange(0, 101, 20))


# Plot 2: Average Healthy Matrix
healthy_plot = axes['Healthy'].imshow(healthy_average_matrix, **imshow_kwargs)
axes['Healthy'].set_title(f'Average FC Matrix - Healthy Controls\n(N={healthy_count})')
axes['Healthy'].set_xlabel('regions')
axes['Healthy'].set_ylabel('regions')
fig.colorbar(healthy_plot, ax=axes['Healthy'], label='Correlation Coefficient')
# Customize tick locations
axes['Healthy'].set_xticks(np.arange(0, 101, 20))
axes['Healthy'].set_yticks(np.arange(0, 101, 20))


# Adjust layout to prevent clipping of titles and labels
plt.tight_layout()

# Save the plot in the same output directory
plot_save_path = os.path.join(output_dir, 'average_fc_matrices.png')
plt.savefig(plot_save_path, dpi=300)
print(f"\nAverage FC matrix plot saved successfully to: {plot_save_path}")

# Display the plot
plt.show()

print("\nAnalysis and plotting complete.")