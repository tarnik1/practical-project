# SCRIPT 02B: FC DIFFERENCE AND SIGNIFICANCE TESTING
# GOAL: Plot the PD - HC difference matrix, and a thresholded significance matrix.

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
from scipy.stats import ttest_ind # <-- IMPORT ADDED FOR T-TEST

# Define the output directory
output_dir = r'C:\Users\nikna\Documents\pp_datasets\processed_data_1'
master_csv_path = os.path.join(output_dir, "master_metadata.csv")

# --- Step 1: Load the Master Metadata ---
try:
    master_df = pd.read_csv(master_csv_path)
    print(f"Loaded master metadata with {len(master_df)} subjects.")
except FileNotFoundError:
    print(f"Error: Could not find master metadata.")
    exit()

# --- Step 2: Initialize Lists to Store ALL Matrices ---
# We must store the individual matrices to calculate the variance for the t-test
pd_matrices = []
healthy_matrices = []

print("\nLoading subjects into memory for statistical testing...")

# --- Step 3: Loop Through Metadata and Collect Matrices ---
for index, row in master_df.iterrows():
    label = row['label']
    matrix_path = row['npy_path']
    subject_id = row['subject_id']

    try:
        fc_matrix = np.load(matrix_path)
        if fc_matrix.shape != (100, 100):
            continue

        if label == 1:
            pd_matrices.append(fc_matrix)
        elif label == 0:
            healthy_matrices.append(fc_matrix)

    except Exception as e:
        print(f"Error processing {subject_id}: {e}")

print(f"  Loaded {len(pd_matrices)} PD subjects and {len(healthy_matrices)} Healthy subjects.")

# --- Step 4: Convert to 3D Numpy Arrays ---
# Shape becomes (Number of Subjects, 100, 100)
pd_stack = np.stack(pd_matrices)
hc_stack = np.stack(healthy_matrices)

# --- Step 5: Calculate Averages and Difference ---
pd_average = np.mean(pd_stack, axis=0)
hc_average = np.mean(hc_stack, axis=0)

# Matrix 1: The Raw Difference (PD minus Healthy)
# Positive values (Red) = Stronger connection in PD
# Negative values (Blue) = Stronger connection in Healthy Controls
diff_matrix = pd_average - hc_average

# --- Step 6: Perform the T-Test ---
print("Calculating T-tests for all 10,000 connections...")
# We use Welch's t-test (equal_var=False) which is safer for uneven group sizes
t_stats, p_values = ttest_ind(pd_stack, hc_stack, axis=0, equal_var=False)

# Define our significance threshold
alpha = 0.05

# Matrix 2: The Significant Difference Matrix
# We create a copy of the difference matrix, but replace any non-significant connection with NaN (Not a Number).
# Matplotlib will naturally plot NaNs as blank/white space.
sig_diff_matrix = np.where(p_values < alpha, diff_matrix, np.nan)

# --- Step 7: Generate the Plots ---
fig, axes = plt.subplots(1, 2, figsize=(18, 7))

# We use 'RdBu_r' (Red-Blue reversed) so Red is positive diff, Blue is negative diff.
# We set vmin and vmax symmetrically so White is always exactly 0.
max_diff = np.max(np.abs(diff_matrix)) 
imshow_kwargs = {'cmap': 'RdBu_r', 'vmin': -max_diff, 'vmax': max_diff}

# Plot 1: Raw Difference
diff_plot = axes[0].imshow(diff_matrix, **imshow_kwargs)
axes[0].set_title(f'Raw FC Difference (PD - Healthy)\nRed = Stronger in PD | Blue = Weaker in PD')
axes[0].set_xlabel('regions')
axes[0].set_ylabel('regions')
fig.colorbar(diff_plot, ax=axes[0], label='Difference in Correlation Coefficient')

# Plot 2: Significant Difference (Masked by p < 0.05)
sig_plot = axes[1].imshow(sig_diff_matrix, **imshow_kwargs)
# We set the background color to a light grey so the white/nan values stand out clearly
axes[1].set_facecolor('#eaeaea') 
axes[1].set_title(f'Significant FC Differences Only\n(Uncorrected p < {alpha})')
axes[1].set_xlabel('regions')
axes[1].set_ylabel('regions')
fig.colorbar(sig_plot, ax=axes[1], label='Difference in Correlation Coefficient')

# Clean up ticks
for ax in axes:
    ax.set_xticks(np.arange(0, 101, 20))
    ax.set_yticks(np.arange(0, 101, 20))

plt.tight_layout()

# Save the plot
plot_save_path = os.path.join(output_dir, 'fc_significance_test.png')
plt.savefig(plot_save_path, dpi=300)
print(f"\nSignificance plot saved to: {plot_save_path}")

plt.show()