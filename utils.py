# HELPER SCRIPT: UTILITIES
# GOAL: Store helper functions and the main PyTorch Dataset class.

# 1. Import libraries
#    (e.g., torch, pandas, numpy)
#    (from torch.utils.data import Dataset)

# 2. Define the PyTorch Dataset Class
#    - class FCDataset(Dataset):
#    -   def __init__(self, metadata_df):
#    -     # ... store the metadata_df
#    -   def __len__(self):
#    -     # ... return len(self.metadata_df)
#    -   def __getitem__(self, idx):
#    -     # ... 1. Get the row for this index from metadata_df
#    -     # ... 2. Load the .npy file from the 'npy_path'
#    -     # ... 3. Get the label ('PD' or 'Control') and convert to 0 or 1
#    -     # ... 4. Convert the FC matrix (adjacency) into PyTorch Geometric format:
#    -     #        - x (node features, can be an identity matrix)
#    -     #        - edge_index (list of connected nodes)
#    -     #        - edge_weight (the FC values)
#    -     # ... 5. Return (data_object, label)

# 3. Define a helper function to load the FAISS index
#    - def load_faiss_index(path):
#    -   # ... import faiss
#    -   # ... read and return the index