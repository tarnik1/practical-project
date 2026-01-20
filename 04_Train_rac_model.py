# SCRIPT 04: TRAIN THE RETRIEVAL-AUGMENTED CLASSIFIER
# GOAL: Train the full RAC model on the TaoWu dataset.

# 1. Import libraries
#    (e.g., torch, pandas, faiss)
#    (from models import GraphAutoencoder, AttentionMechanism, RAC_Model)
#    (from utils import FCDataset, load_faiss_index)

import os
import torch
import pandas as pd
import numpy as np
import faiss
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
from models import GraphAutoencoder, AttentionMechanism, RAC_Model
from utils import FCDataset

output_dir = r'C:\Users\nikna\Documents\pp_datasets\processed_data_1'
master_csv = os.path.join(output_dir, "master_metadata.csv")
encoder_weights = os.path.join(output_dir, 'gae_encoder.pth')
index_path = os.path.join(output_dir, 'knowledge_base.index')
kb_embeddings_path = os.path.join(output_dir, 'kb_embeddings.npy') # The "Shelf"
kb_metadata_path = os.path.join(output_dir, 'kb_metadata_indexed.csv')

# 2. Load the metadata and filter for the *target* dataset
#    - (e.g., df_target = df[df['dataset_source'] == 'TaoWu'])
#    - Split df_target into df_train and df_val (e.g., 80/20 split)

# LOAD & SPLIT TARGET DATA (TaoWu)
df = pd.read_csv(master_csv)
df_target = df[df['dataset_source'] == 'TaoWu'].reset_index(drop=True)

# 80/20 split for training the Attention/Classification layers
df_train, df_val = train_test_split(df_target, test_size=0.2, stratify=df_target['label'], random_state=42)

# 3. Create PyTorch Datasets and DataLoaders for train and validation
#    - train_dataset = FCDataset(df_train)
#    - train_dataloader = DataLoader(train_dataset, ...)

# DATALOADERS
train_loader = DataLoader(FCDataset(df_train), batch_size=16, shuffle=True)
val_loader = DataLoader(FCDataset(df_val), batch_size=16, shuffle=False)

# 4. Load the pre-trained GAE encoder
#    - gae_encoder = GraphAutoencoder(...).encoder
#    - gae_encoder.load_state_dict(torch.load('gae_encoder.pth'))

# 4. LOAD PRE-TRAINED GAE ENCODER
# Using the structure from our discussion
base_model = GraphAutoencoder(num_nodes=100, input_dim=100, hidden_dim=64, embedding_dim=128)
base_model.encoder.load_state_dict(torch.load(encoder_weights))
gae_encoder = base_model.encoder
gae_encoder.eval() # Keep encoder frozen initially

# 5. Load the FAISS index
#    - index = load_faiss_index('knowledge_base.index')
#    - K = 5 # (Number of neighbors to retrieve)

# 5. LOAD FAISS INDEX & KB EMBEDDINGS
index = faiss.read_index(index_path)
kb_embeddings = np.load(kb_embeddings_path).astype('float32')
K = 5 # Number of neighbors

# 6. Initialize the full RAC model
#    - attention_model = AttentionMechanism(...)
#    - rac_model = RAC_Model(gae_encoder, attention_model, ...)
#    - optimizer = torch.optim.Adam(rac_model.parameters(), lr=0.001)
#    - loss_fn = torch.nn.BCEWithLogitsLoss() # (For classification)

# 6. INITIALIZE RAC MODEL
attention = AttentionMechanism(embedding_dim=128)
rac_model = RAC_Model(gae_encoder, attention, embedding_dim=128)

optimizer = torch.optim.Adam(rac_model.parameters(), lr=0.001)
# Using BCELoss because our RAC_Model ends with a Sigmoid
loss_fn = torch.nn.BCELoss()

# 7. Start the training loop
#    - for epoch in range(num_epochs):
#    -   for (data, label) in train_dataloader:
#    -     # ... 1. Get the query vector (with no gradient)
#    -     #      with torch.no_grad():
#    -     #        v_query = gae_encoder(data)
#    -     # ... 2. Retrieve neighbors from FAISS (this is a CPU/Numpy step)
#    -     #      distances, indices = index.search(v_query.cpu().numpy(), K)
#    -     # ... 3. Get the actual retrieved vectors (V_retrieved)
#    -     #      (This is complex: you need to map 'indices' back to your
#    -     #      'all_embeddings' from script 03.
#    -     #      It's best to save 'all_embeddings' as a .npy file)
#    -     # ... 4. Pass everything to the model
#    -     #      prediction = rac_model(data, V_retrieved)
#    -     # ... 5. Calculate loss (loss = loss_fn(prediction, label))
#    -     # ... 6. Backpropagate and update weights
#    -   # ... Run a similar loop on the validation set to check performance

num_epochs = 20
for epoch in range(num_epochs):
    rac_model.train()
    total_loss = 0
    
    for data, labels in train_loader:
        optimizer.zero_grad()
        
        # Step A: Get Query Vector (from GAE)
        with torch.no_grad():
            v_query = gae_encoder(data.x, data.edge_index, data.edge_weight, data.batch)
        
        # Step B: Retrieve Neighbors from FAISS (Numpy step)
        v_query_np = v_query.cpu().numpy().astype('float32')
        distances, neighbor_indices = index.search(v_query_np, K)
        
        # Step C: Map indices back to the actual vectors (V_retrieved)
        # neighbor_indices is (Batch, K), we use it to slice our .npy array
        v_retrieved = torch.from_numpy(kb_embeddings[neighbor_indices]).to(v_query.device)
        
        # Step D: Forward Pass
        # We pass the data AND the neighbors we just found
        predictions, _ = rac_model(data.x, data.edge_index, data.edge_weight, data.batch, v_retrieved)
        
        # Step E: Loss & Backprop
        loss = loss_fn(predictions.squeeze(), labels.float())
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()

    print(f"Epoch {epoch+1}/{num_epochs}, Loss: {total_loss/len(train_loader):.4f}")

# 8. Save the final trained RAC model
#    - (e.g., torch.save(rac_model.state_dict(), 'rac_model.pth'))

torch.save(rac_model.state_dict(), os.path.join(output_dir, 'rac_model.pth'))
print("RAC Model Training Complete and Saved.")