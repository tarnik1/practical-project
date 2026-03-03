# SCRIPT 05: TRAIN THE BASELINE (AB-INITIO) MODEL
# GOAL: Train a simple GNN on the TaoWu dataset from scratch.

# 1. Import libraries
#    (e.g., torch, pandas)
#    (from models import GraphAutoencoder, Baseline_GNN)
#    (from utils import FCDataset)

import os
import torch
import pandas as pd
from torch_geometric.loader import DataLoader
from sklearn.model_selection import train_test_split
from models import GraphAutoencoder, Baseline_GNN
from utils import FCDataset

output_dir = r'C:\Users\nikna\Documents\pp_datasets\processed_data_1'
master_csv = os.path.join(output_dir, "master_metadata.csv")

# 2. Load metadata, create train/val splits for TaoWu
#    - (Same as step 2 & 3 in script 04)

df = pd.read_csv(master_csv)
df_target = df[df['dataset_source'] == 'TaoWu'].reset_index(drop=True)
df_train, df_val = train_test_split(df_target, test_size=0.2, stratify=df_target['label'], random_state=42)

train_loader = DataLoader(FCDataset(df_train), batch_size=16, shuffle=True)
val_loader = DataLoader(FCDataset(df_val), batch_size=16, shuffle=False)

# 3. Initialize the Baseline model
#    - # Note: We initialize a NEW encoder, not the pre-trained one, new encoder/fresh random weights
#    - gae_encoder_scratch = GraphAutoencoder(...).encoder
#    - baseline_model = Baseline_GNN(gae_encoder_scratch, ...)
#    - optimizer = torch.optim.Adam(baseline_model.parameters(), lr=0.001)
#    - loss_fn = torch.nn.BCEWithLogitsLoss()

fresh_gae = GraphAutoencoder(num_nodes=100, input_dim=100, hidden_dim=64, embedding_dim=128)
# we then pass the entire fresh_gae into the Baseline_GNN
baseline_model = Baseline_GNN(gae_encoder=fresh_gae, embedding_dim=128)

optimizer = torch.optim.Adam(baseline_model.parameters(), lr=0.001)

# Since the Baseline_GNN model ends with a Sigmoid, i use BCELoss.
# without Sigmoid, i would use BCEWithLogitsLoss for better numerical stability.
loss_fn = torch.nn.BCELoss()

# 4. Start the training loop
#    - for epoch in range(num_epochs):
#    -   for (data, label) in train_dataloader:
#    -     # ... 1. Get prediction (prediction = baseline_model(data))
#    -     # ... 2. Calculate loss
#    -     # ... 3. Backpropagate and update
#    -   # ... Run validation loop

num_epochs = 20
print("Starting Baseline Model Training on TaoWu...")

for epoch in range(num_epochs):
    baseline_model.train()
    total_train_loss = 0
    for data in train_loader:
        optimizer.zero_grad()
        
        # Forward pass (Only target data, no retrieval)
        predictions = baseline_model(data.x, data.edge_index, data.edge_weight, data.batch)
        loss = loss_fn(predictions.squeeze(), data.y.float().squeeze())
        loss.backward()
        optimizer.step()
        
        total_train_loss += loss.item()

    baseline_model.eval()
    total_val_loss = 0
    correct = 0
    total = 0
    
    with torch.no_grad():
        for val_data in val_loader:
            preds = baseline_model(val_data.x, val_data.edge_index, val_data.edge_weight, val_data.batch)
            val_loss = loss_fn(preds.squeeze(), val_data.y.float().squeeze())
            
            total_val_loss += val_loss.item()
            
            # Calculate accuracy: If prediction > 0.5, guess PD (1), else guess Control (0)
            predicted_class = (preds.squeeze() > 0.5).float()
            correct += (predicted_class == val_data.y.float().squeeze()).sum().item()
            total += val_data.y.size(0)

    avg_train = total_train_loss / len(train_loader)
    avg_val = total_val_loss / len(val_loader)
    accuracy = correct / total
    
    print(f"Epoch {epoch+1}/{num_epochs}")
    print(f"Train Loss: {avg_train:.4f} | Val Loss: {avg_val:.4f} | Val Acc: {accuracy:.2%}")

# 5. Save the trained baseline model
#    - (e.g., torch.save(baseline_model.state_dict(), 'baseline_model.pth'))

torch.save(baseline_model.state_dict(), os.path.join(output_dir, 'baseline_model.pth'))
print("Baseline Model (Ab-Initio) Training Complete and Saved.")