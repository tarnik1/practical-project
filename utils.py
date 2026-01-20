# HELPER SCRIPT: UTILITIES
# GOAL: Store helper functions and the main PyTorch Dataset class.

# the "Data" object must contain 3 specific elements:
# 1. x (node features): what does each (of the 100) brain regons look like?
# 2. edge_index (Topology): which brain regions are connected to which?
# 3. edge_weight (Strength): how strong is the correlation (the actual values from the FC matrix)?

# 1. Import libraries
#    (e.g., torch, pandas, numpy)
#    (from torch.utils.data import Dataset)

import torch
import numpy as np
import pandas as pd
from torch.utils.data import Dataset
from torch_geometric.data import Data

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

class FCDataset(Dataset):
    def __init__(self, metadata_df):
        self.metadata_df = metadata_df

    def __len__(self):
        return len(self.metadata_df)
    
    def __getitem__(self, idx):
        row = self.metadata_df.iloc[idx]
        
        # Loading the .npy file
        matrix = np.load(row['npy_path']) # using the npy_path column from script 01 to locate and load the FC matrix
        
        # Converting label
        diag = str(row['diagnosis']).lower()
        label = 1 if 'pd' in diag else 0
        
        # Converting FC matrix into PyTorch Geometric format
        num_nodes = matrix.shape[0] # should be 100
        
        # Identitying matrix for node features (Noman et al.)
        x = torch.eye(num_nodes, dtype=torch.float) # torch.eye() returns a 2-D tensor of size (num_nodes, num_nodes) with ones on the diagonal and zeros elsewhere.
        
        # Creating edge_index (every node connected to every node)
        adj = np.ones((num_nodes, num_nodes))
        edge_index = torch.tensor(np.array(np.nonzero(adj)), dtype=torch.long) # a list of every possible connection in the brain
        # edge_index has shape [2, 10000 (number of edges)]. row 0: source nodes, row 1: target nodes.
        
        # Using matrix values as edge weights
        edge_weight = torch.tensor(matrix.flatten(), dtype=torch.float) # flattening the 100*100 matrix into 10000 values.
        # into the model => it will represent the strength of connection between the nodes/brain regions.
        
        # Creating the Data object
        data_obj = Data(x=x, edge_index=edge_index, edge_weight=edge_weight)
        data_obj.y = torch.tensor(matrix, dtype=torch.float) # Target for reconstruction =>
        # storing the original FC matrix here; the "ground Truth" that the GAE will try to reconstruct during training.
        
        return data_obj, label

# 3. Define a helper function to load the FAISS index
#    - def load_faiss_index(path):
#    -   # ... import faiss
#    -   # ... read and return the index

def load_faiss_index(path): # come back and change path later
    import faiss
    return faiss.read_index(path)