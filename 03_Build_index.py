# SCRIPT 03: BUILD FAISS INDEX
# GOAL: Create and save the FAISS index from the knowledge base embeddings.

# 1. Import libraries
#    (e.g., torch, pandas, faiss, numpy)
#    (from models import GraphAutoencoder, from utils import FCDataset)

import os
import torch
import pandas as pd
import numpy as np
import faiss
from torch_geometric.loader import DataLoader
from models import GraphAutoencoder
from utils import FCDataset

output_dir = r'C:\Users\nikna\Documents\pp_datasets\processed_data_1'
master_csv = os.path.join(output_dir, "master_metadata.csv")
encoder_weights = os.path.join(output_dir, 'gae_encoder.pth')
index_save_path = os.path.join(output_dir, 'knowledge_base.index')

# 2. Load the metadata and filter for the knowledge base (like in script 02).
#    - df_kb = ...

df = pd.read_csv(master_csv)
df_kb = df[df['dataset_source'].isin(['PPMI', 'Neurocon'])].reset_index(drop=False)
df_kb = df_kb.rename(columns={'index': 'original_master_index'})

# 3. Create a DataLoader (no shuffling needed)
#    - kb_dataset = FCDataset(df_kb)
#    - kb_dataloader = DataLoader(kb_dataset, batch_size=32, shuffle=False)

kb_dataset = FCDataset(df_kb)
kb_dataloader = DataLoader(kb_dataset, batch_size=32, shuffle=False)

# 4. Initialize the encoder and load its trained weights
#    - encoder = GraphAutoencoder(...).encoder
#    - encoder.load_state_dict(torch.load('gae_encoder.pth'))
#    - encoder.eval() # Set to evaluation mode

model = GraphAutoencoder(num_nodes=100, input_dim=100, hidden_dim=64, embedding_dim=128)
model.encoder.load_state_dict(torch.load(encoder_weights))
model.encoder.eval()

# 5. Loop through the dataloader and collect all embeddings
#    - all_embeddings = []
#    - with torch.no_grad():
#    -   for (data, label) in kb_dataloader:
#    -     # ... v_embedding = encoder(data)
#    -     # ... all_embeddings.append(v_embedding.cpu().numpy())
#    - all_embeddings = np.concatenate(all_embeddings, axis=0)

print("encoding Knowledge Base into vector space...")
all_embeddings = []

with torch.no_grad(): # disables gradient calculations to save memory and computation
    for data in kb_dataloader:
        
        v_embedding = model.encoder(data.x, data.edge_index, data.edge_weight, data.batch) # without passing data.batch, the encoder wouldn't know where one brain ends and the next begins.
        all_embeddings.append(v_embedding.cpu().numpy().astype('float32'))

all_embeddings = np.concatenate(all_embeddings, axis=0) # FAISS needs float32

# 6. Create and populate the FAISS index
#    - EMBEDDING_DIM = 128
#    - index = faiss.IndexFlatL2(EMBEDDING_DIM) # (L2 distance = dot product for normalized vectors)
#    - index.add(all_embeddings)

EMBEDDING_DIM = 128
index = faiss.IndexFlatL2(EMBEDDING_DIM) # (L2 distance = dot product for normalized vectors)
index.add(all_embeddings)
# "IndexFlatL2" tells FAISS to store vectors as-is, without compressing or clustering them.
# since the knowledge base isn't too large, this is a good method because it performs an exact search rather than an approximation.
# L2 => Euclidean distance

# 7. Save the index to disk
#    - (e.g., faiss.write_index(index, 'knowledge_base.index'))

faiss.write_index(index, index_save_path)
np.save(os.path.join(output_dir, 'kb_embeddings.npy'), all_embeddings)
df_kb.to_csv(os.path.join(output_dir, "kb_metadata_indexed.csv"), index=False)
# we save this to know which row in the KB corresponds to which embedding in FAISS.

print(f"Knowledge Base Index saved with {index.ntotal} subjects.")
print(f"Embeddings saved to: kb_embeddings.npy")
print(f"Index successfully saved to: {output_dir}")