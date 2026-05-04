import numpy as np

def inspect_latent_corruption(emb_path, label_path):
    # 1. Load data
    try:
        X = np.load(emb_path)
        y = np.load(label_path)
    except FileNotFoundError:
        print("Files not found. Ensure paths are correct.")
        return

    # 2. Identify the locations of NaNs and Infs
    is_bad = np.isnan(X) | np.isinf(X)
    
    # 3. Find indices of problematic SUBJECTS (rows)
    # A subject is bad if ANY dimension in their 128-vec is NaN/Inf
    bad_subject_indices = np.where(is_bad.any(axis=1))[0]
    
    # 4. Find indices of problematic DIMENSIONS (columns)
    # A dimension is bad if ANY subject has a NaN/Inf in that specific slot
    bad_dim_indices = np.where(is_bad.any(axis=0))[0]

    print("="*40)
    print(f"CORRUPTION REPORT for: {emb_path}")
    print("="*40)

    if len(bad_subject_indices) == 0:
        print("✅ SUCCESS: No NaNs or Infs found in the embedding space.")
        print(f"Mean of latent space: {np.mean(X):.4f}")
        print(f"Std of latent space:  {np.std(X):.4f}")
    else:
        print(f"❌ TOTAL CORRUPTED SUBJECTS: {len(bad_subject_indices)} / {X.shape[0]}")
        print(f"Subject Indices: {bad_subject_indices.tolist()}")
        
        print(f"\n❌ TOTAL CORRUPTED DIMENSIONS: {len(bad_dim_indices)} / {X.shape[1]}")
        print(f"Dimension Indices: {bad_dim_indices.tolist()}")
        
        # Calculate Percentage of Failure
        fail_percent = (len(bad_subject_indices) / X.shape[0]) * 100
        print(f"\nFailure Rate: {fail_percent:.2f}%")
        
        # Check if one specific label is failing (e.g., only Patients)
        bad_labels = y[bad_subject_indices]
        unique, counts = np.unique(bad_labels, return_counts=True)
        label_map = {0: "Control", 1: "Patient"}
        print("\nCorruption by Class:")
        for u, c in zip(unique, counts):
            print(f" - {label_map[u]}: {c} subjects affected")

    print("="*40)

# Run the inspection
inspect_latent_corruption('pd_hc_embeddings.npy', 'pd_hc_labels.npy')