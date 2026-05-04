# SCRIPT 05: TRAIN THE BASELINE (AB-INITIO) MODEL (5-Fold CV)
# GOAL: Train a simple GNN on the TaoWu dataset from scratch using 5-Fold Cross Validation.

import os
import torch
import pandas as pd
import numpy as np
from torch_geometric.loader import DataLoader
from sklearn.model_selection import StratifiedKFold
from models import GraphAutoencoder, Baseline_GNN
from utils import FCDataset

output_dir = r'C:\Users\nikna\Documents\pp_datasets\processed_data_1'
master_csv = os.path.join(output_dir, "master_metadata.csv")

# 2. Load metadata and filter for the target dataset
df = pd.read_csv(master_csv)
df_target = df[df['dataset_source'] == 'TaoWu'].reset_index(drop=True)

# 3. Setup 5-Fold Cross Validation
n_splits = 5
skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)

fold_mean_accuracies = []
fold_final_accuracies = []

print(f"Starting {n_splits}-Fold Cross Validation for Baseline Model on TaoWu...")

for fold, (train_idx, val_idx) in enumerate(skf.split(df_target, df_target['label'])):
    print(f"\n==========================================")
    print(f"               FOLD {fold + 1}/{n_splits}               ")
    print(f"==========================================")
    
    # Split data for this fold
    df_train = df_target.iloc[train_idx].reset_index(drop=True)
    df_val = df_target.iloc[val_idx].reset_index(drop=True)

    train_loader = DataLoader(FCDataset(df_train), batch_size=16, shuffle=True)
    val_loader = DataLoader(FCDataset(df_val), batch_size=16, shuffle=False)

    # RE-INITIALIZE MODEL FOR EACH FOLD (Total Amnesia)
    # We initialize a NEW encoder with fresh random weights
    fresh_gae = GraphAutoencoder(num_nodes=100, input_dim=100, hidden_dim=64, embedding_dim=128)
    baseline_model = Baseline_GNN(gae_encoder=fresh_gae.encoder, embedding_dim=128)

    optimizer = torch.optim.Adam(baseline_model.parameters(), lr=0.001)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=3)
    loss_fn = torch.nn.BCELoss()

    num_epochs = 100
    all_val_accuracies = []

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
        
        correct_preds = 0
        total_samples = 0
        
        with torch.no_grad():
            for val_data in val_loader:
                preds = baseline_model(val_data.x, val_data.edge_index, val_data.edge_weight, val_data.batch)
                
                val_loss = loss_fn(preds.squeeze(), val_data.y.float().squeeze())
                total_val_loss += val_loss.item()
        
                binary_preds = (preds.squeeze() > 0.5).float()
                true_labels = val_data.y.float().squeeze()
                
                # Safety check for batch size of 1
                if true_labels.dim() == 0:
                    true_labels = true_labels.unsqueeze(0)
                    binary_preds = binary_preds.unsqueeze(0)
                    
                correct_preds += (binary_preds == true_labels).sum().item()
                total_samples += true_labels.size(0)
        
        epoch_val_acc = correct_preds / total_samples
        all_val_accuracies.append(epoch_val_acc)

        avg_val_loss = total_val_loss / len(val_loader)
        scheduler.step(avg_val_loss)
        
        if (epoch + 1) % 10 == 0:
            print(f"Epoch {epoch+1:02d}/{num_epochs} | Train Loss: {total_train_loss/len(train_loader):.4f} | Val Loss: {total_val_loss/len(val_loader):.4f} | Val Acc: {epoch_val_acc * 100:.2f}%")

    # Save the trained baseline model for this specific fold
    torch.save(baseline_model.state_dict(), os.path.join(output_dir, f'baseline_model_fold_{fold+1}.pth'))
    
    # Calculate fold metrics
    mean_val_acc = np.mean(all_val_accuracies)
    final_val_acc = all_val_accuracies[-1]
    
    fold_mean_accuracies.append(mean_val_acc)
    fold_final_accuracies.append(final_val_acc)
    
    print(f"\n--> Fold {fold + 1} Complete!")
    print(f"--> Fold {fold + 1} Mean Accuracy: {mean_val_acc * 100:.2f}%")
    print(f"--> Fold {fold + 1} Final Epoch Accuracy: {final_val_acc * 100:.2f}%\n")

# Overall Results
overall_mean_acc = np.mean(fold_mean_accuracies) * 100
overall_final_acc = np.mean(fold_final_accuracies) * 100
overall_final_std = np.std(fold_final_accuracies) * 100

print("--------------------------------------------------")
print("5-FOLD CV BASELINE (AB-INITIO) TRAINING COMPLETE.")
print(f"OVERALL Mean Validation Accuracy: {overall_mean_acc:.2f}%")
print(f"OVERALL Final Epoch Accuracy: {overall_final_acc:.2f}% ± {overall_final_std:.2f}%")
print("--------------------------------------------------")