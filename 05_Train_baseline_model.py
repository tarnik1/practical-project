# SCRIPT 05: TRAIN THE BASELINE (AB-INITIO) MODEL
# GOAL: Train a simple GNN on the TaoWu dataset from scratch.

# 1. Import libraries
#    (e.g., torch, pandas)
#    (from models import GraphAutoencoder, Baseline_GNN)
#    (from utils import FCDataset)

import os
import torch
import pandas as pd
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
from models import GraphAutoencoder, Baseline_GNN
from utils import FCDataset

output_dir = r'C:\Users\nikna\Documents\pp_datasets\processed_data_1'
master_csv = os.path.join(output_dir, "master_metadata.csv")

# 2. Load metadata, create train/val splits for TaoWu
#    - (Same as step 2 & 3 in script 04)

# 2. LOAD & SPLIT DATA (Same as Script 04 for consistency)
df = pd.read_csv(master_csv)
df_target = df[df['dataset_source'] == 'TaoWu'].reset_index(drop=True)

# 80/20 split - Use the same random_state=42 as Script 04
df_train, df_val = train_test_split(df_target, test_size=0.2, stratify=df_target['label'], random_state=42)

train_loader = DataLoader(FCDataset(df_train), batch_size=16, shuffle=True)
val_loader = DataLoader(FCDataset(df_val), batch_size=16, shuffle=False)

# 3. Initialize the Baseline model
#    - # Note: We initialize a NEW encoder, not the pre-trained one
#    - gae_encoder_scratch = GraphAutoencoder(...).encoder
#    - baseline_model = Baseline_GNN(gae_encoder_scratch, ...)
#    - optimizer = torch.optim.Adam(baseline_model.parameters(), lr=0.001)
#    - loss_fn = torch.nn.BCEWithLogitsLoss()

# 3. INITIALIZE THE BASELINE MODEL (FROM SCRATCH)
# We initialize a NEW GraphAutoencoder to get a "fresh" encoder with random weights
fresh_gae = GraphAutoencoder(num_nodes=100, input_dim=100, hidden_dim=64, embedding_dim=128)
baseline_model = Baseline_GNN(gae_encoder=fresh_gae.encoder, embedding_dim=128)

optimizer = torch.optim.Adam(baseline_model.parameters(), lr=0.001)

# Note: Since the Baseline_GNN model ends with a Sigmoid, we use BCELoss.
# If you removed Sigmoid, you would use BCEWithLogitsLoss for better numerical stability.
loss_fn = torch.nn.BCELoss()

# 4. Start the training loop
#    - for epoch in range(num_epochs):
#    -   for (data, label) in train_dataloader:
#    -     # ... 1. Get prediction (prediction = baseline_model(data))
#    -     # ... 2. Calculate loss
#    -     # ... 3. Backpropagate and update
#    -   # ... Run validation loop

num_epochs = 20
for epoch in range(num_epochs):
    # --- TRAINING PHASE ---
    baseline_model.train()
    total_train_loss = 0
    for data, labels in train_loader:
        optimizer.zero_grad()
        
        # Forward pass (Only target data, no retrieval)
        predictions = baseline_model(data.x, data.edge_index, data.edge_weight, data.batch)
        
        loss = loss_fn(predictions.squeeze(), labels.float())
        loss.backward()
        optimizer.step()
        total_train_loss += loss.item()

    # --- VALIDATION PHASE ---
    baseline_model.eval()
    total_val_loss = 0
    correct = 0
    total = 0
    with torch.no_grad():
        for data, labels in val_loader:
            preds = baseline_model(data.x, data.edge_index, data.edge_weight, data.batch)
            val_loss = loss_fn(preds.squeeze(), labels.float())
            total_val_loss += val_loss.item()
            
            # Calculate accuracy
            predicted_class = (preds > 0.5).float()
            correct += (predicted_class.squeeze() == labels).sum().item()
            total += labels.size(0)

    avg_train = total_train_loss / len(train_loader)
    avg_val = total_val_loss / len(val_loader)
    accuracy = correct / total
    
    print(f"Epoch {epoch+1}/{num_epochs}")
    print(f"Train Loss: {avg_train:.4f} | Val Loss: {avg_val:.4f} | Val Acc: {accuracy:.2%}")

# 5. Save the trained baseline model
#    - (e.g., torch.save(baseline_model.state_dict(), 'baseline_model.pth'))

torch.save(baseline_model.state_dict(), os.path.join(output_dir, 'baseline_model.pth'))
print("Baseline Model (Ab-Initio) Training Complete.")