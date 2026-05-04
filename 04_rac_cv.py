# SCRIPT 04: TRAIN THE RETRIEVAL-AUGMENTED CLASSIFIER
# GOAL: Train the full RAC model on the TaoWu dataset with Leave-One-Out Cross Validation.

import os
import torch
import pandas as pd
import numpy as np
import faiss
from torch_geometric.loader import DataLoader
from sklearn.model_selection import LeaveOneOut
from sklearn.metrics import roc_auc_score
from models import GraphAutoencoder, AttentionMechanism, RAC_Model
from utils import FCDataset

output_dir = r'C:\Users\nikna\Documents\pp_datasets\processed_data_1'
master_csv = os.path.join(output_dir, "master_metadata.csv")
encoder_weights = os.path.join(output_dir, 'gae_encoder.pth')
index_path = os.path.join(output_dir, 'knowledge_base.index')
kb_embeddings_path = os.path.join(output_dir, 'kb_embeddings.npy')
kb_metadata_path = os.path.join(output_dir, 'kb_metadata_indexed.csv')

df = pd.read_csv(master_csv)
df_target = df[df['dataset_source'] == 'TaoWu'].reset_index(drop=True)

model = GraphAutoencoder(num_nodes=100, input_dim=100, hidden_dim=64, embedding_dim=128)
model.encoder.load_state_dict(torch.load(encoder_weights))
model.encoder.eval()
for param in model.encoder.parameters():
    param.requires_grad = False

index = faiss.read_index(index_path)
kb_embeddings = np.load(kb_embeddings_path).astype('float32')
K = 10

loo = LeaveOneOut()
all_labels = []
all_predictions = []
all_probs = []

print(f"Starting Leave-One-Out Cross Validation ({len(df_target)} folds)...")

for fold, (train_idx, val_idx) in enumerate(loo.split(df_target)):
    df_train = df_target.iloc[train_idx].reset_index(drop=True)
    df_val = df_target.iloc[val_idx].reset_index(drop=True)

    train_loader = DataLoader(FCDataset(df_train), batch_size=16, shuffle=True)
    val_loader = DataLoader(FCDataset(df_val), batch_size=1, shuffle=False)

    # Re-initialize RAC model fresh for each fold
    attention_model = AttentionMechanism(embedding_dim=128)
    rac_model = RAC_Model(model.encoder, attention_model, embedding_dim=128)

    optimizer = torch.optim.Adam(rac_model.parameters(), lr=0.001)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=3)
    loss_fn = torch.nn.BCELoss()

    # Training
    num_epochs = 50
    for epoch in range(num_epochs):
        rac_model.train()
        total_loss = 0

        for data in train_loader:
            optimizer.zero_grad()

            with torch.no_grad():
                v_query = model.encoder(data.x, data.edge_index, data.edge_weight, data.batch)

            v_query_np = v_query.cpu().numpy().astype('float32')
            _, neighbor_indices = index.search(v_query_np, K)
            v_retrieved = torch.from_numpy(kb_embeddings[neighbor_indices]).to(v_query.device)

            predictions, _ = rac_model(data.x, data.edge_index, data.edge_weight, data.batch, v_retrieved)
            loss = loss_fn(predictions.squeeze(), data.y.float().squeeze())
            loss.backward()
            torch.nn.utils.clip_grad_norm_(rac_model.parameters(), max_norm=1.0)
            optimizer.step()
            total_loss += loss.item()

        scheduler.step(total_loss / len(train_loader))

    # Evaluation on the single left-out subject
    rac_model.eval()
    with torch.no_grad():
        for val_data in val_loader:
            val_query = model.encoder(val_data.x, val_data.edge_index, val_data.edge_weight, val_data.batch)
            val_query_np = val_query.cpu().numpy().astype('float32')
            _, val_neighbor_indices = index.search(val_query_np, K)
            val_retrieved = torch.from_numpy(kb_embeddings[val_neighbor_indices]).to(val_query.device)

            val_predictions, _ = rac_model(val_data.x, val_data.edge_index, val_data.edge_weight, val_data.batch, val_retrieved)
            prob = val_predictions.squeeze().item()
            predicted_label = 1 if prob > 0.5 else 0
            true_label = int(val_data.y.squeeze().item())

            all_probs.append(prob)
            all_predictions.append(predicted_label)
            all_labels.append(true_label)

    if (fold + 1) % 10 == 0:
        print(f"Fold {fold + 1}/{len(df_target)} complete | True: {true_label} | Pred: {predicted_label} | Prob: {prob:.3f}")
    if (fold + 1) % 10 == 0:
        print(f"Fold {fold + 1}/{len(df_target)} complete | Last Train Loss: {total_loss/len(train_loader):.4f} | True: {true_label} | Pred: {predicted_label} | Prob: {prob:.3f}")

# Final metrics across all folds
all_labels = np.array(all_labels)
all_predictions = np.array(all_predictions)
all_probs = np.array(all_probs)

accuracy = np.mean(all_predictions == all_labels) * 100
auc = roc_auc_score(all_labels, all_probs)

print(f"\n======================================")
print(f"LOO-CV ACCURACY: {accuracy:.2f}%")
print(f"LOO-CV AUC-ROC:  {auc:.4f}")
print(f"======================================")

# Save the last fold's model as rac_cv
torch.save(rac_model.state_dict(), os.path.join(output_dir, 'rac_cv.pth'))
print("RAC CV Model saved as rac_cv.pth")