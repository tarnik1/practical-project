# SCRIPT 02: TRAIN GRAPH AUTOENCODER
# GOAL: Train the GAE on the knowledge base (PPMI, Neurocon).

# 1. Import libraries
#    (e.g., torch, pandas, and from models import GraphAutoencoder, from utils import FCDataset)

import os
import torch
import pandas as pd
from torch_geometric.loader import DataLoader
from models import GraphAutoencoder
from utils import FCDataset

# 2. Load the master metadata.csv
#    - (e.g., df = pd.read_csv(OUTPUT_DIR / 'metadata.csv'))

output_dir = r'C:\Users\nikna\Documents\pp_datasets\processed_data_1'
master_csv = os.path.join(output_dir, "master_metadata.csv")
df = pd.read_csv(master_csv)

# 3. Filter the metadata for the knowledge base
#    - (e.g., df_kb = df[df['dataset_source'].isin(['PPMI', 'Neurocon'])])

df_kb = df[df['dataset_source'].isin(['PPMI', 'Neurocon'])].reset_index(drop=False)
df_kb = df_kb.rename(columns={'index': 'original_master_index'})
print(f"Total Knowledge Base samples: {len(df_kb)}")

# 4. Create PyTorch Dataset and DataLoader
#    - kb_dataset = FCDataset(df_kb)
#    - kb_dataloader = DataLoader(kb_dataset, batch_size=32, shuffle=True)

kb_dataset = FCDataset(df_kb)
kb_dataloader = DataLoader(kb_dataset, batch_size=32, shuffle=True) # 32 subjects/graphs per batch

# 5. Initialize the GAE model, optimizer, and loss function
#    - model = GraphAutoencoder(...)
#    - optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
#    - loss_fn = torch.nn.MSELoss()  # (Reconstruction loss)

model = GraphAutoencoder(num_nodes=100, input_dim=100, hidden_dim=64, embedding_dim=128)
optimizer = torch.optim.Adam(model.parameters(), lr=0.0001, weight_decay=1e-5) # learning rate
loss_fn = torch.nn.MSELoss()

# 6. Start the training loop
#    - for epoch in range(num_epochs):
#    -   for (data, label) in kb_dataloader:
#    -     # ... 1. Zero gradients (optimizer.zero_grad())
#    -     # ... 2. Get model output (reconstructed_matrix = model(data))
#    -     # ... 3. Calculate loss (loss = loss_fn(reconstructed_matrix, original_matrix))
#    -     # ... 4. Backpropagate (loss.backward())
#    -     # ... 5. Update weights (optimizer.step())
#    -   # ... Print epoch loss

num_epochs = 100
# an epoch is one complete pass of the training algorithm through the entire training dataset.
# num_epochs = 50 means the model will see every graph in the knowledge base 50 times during this training.
# in the first few epochs, the model usually just learns the basic shape of the brain.
# in Nomen et al., a range of 50-200 epochs was used (i guess?).
# for consideration: if the loss is still dropping rapidly at epoch 50, increase the number. if the loss stops moving at epoch 30, then 50 was more than enough (you could decrease it).
print("Starting GAE Training...")

for epoch in range(num_epochs):
    model.train()
    total_loss = 0
    for data in kb_dataloader:
        optimizer.zero_grad()

        # Forward pass: Matrix -> Embedding -> Reconstruction
        reconstructed_matrix = model(data.x, data.edge_index, data.edge_weight, data.batch)
        
        # Calculate reconstruction error against original matrix
        loss = loss_fn(reconstructed_matrix, data.fc_matrix)
        loss.backward()

        # to prevent gradients from exploding past a maximum value of 1.0
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        
        optimizer.step()
        total_loss += loss.item() # this is the sum of every mistake the model makes during each entire epoch.
    
    if (epoch + 1) % 10 == 0:
        print(f"Epoch {epoch+1:03d} | Loss: {total_loss/len(kb_dataloader):.6f}")

# consider printing the loss for every epoch: i did. it looks like it already learns most of it by epoch 15.

# 7. Save the trained *encoder* part of the GAE
#    - (e.g., torch.save(model.encoder.state_dict(), 'gae_encoder.pth'))

encoder_path = os.path.join(output_dir, 'gae_encoder.pth')
torch.save(model.encoder.state_dict(), encoder_path)

print(f"GAE Training Complete. Encoder weights saved to: {encoder_path}")