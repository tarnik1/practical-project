# SCRIPT 04: TRAIN THE RETRIEVAL-AUGMENTED CLASSIFIER
# GOAL: Train the full RAC model on the TaoWu dataset.

import os
import torch
import pandas as pd
import numpy as np
import faiss
from torch_geometric.loader import DataLoader
from sklearn.model_selection import train_test_split
from models import GraphAutoencoder, AttentionMechanismLinear, RAC_Model
from utils import FCDataset

output_dir = r'C:\Users\nikna\Documents\pp_datasets\processed_data_1'
master_csv = os.path.join(output_dir, "master_metadata.csv")
encoder_weights = os.path.join(output_dir, 'gae_encoder.pth')
index_path = os.path.join(output_dir, 'knowledge_base.index')
kb_embeddings_path = os.path.join(output_dir, 'kb_embeddings.npy')
kb_metadata_path = os.path.join(output_dir, 'kb_metadata_indexed.csv')

df = pd.read_csv(master_csv)
df_target = df[df['dataset_source'] == 'TaoWu'].reset_index(drop=True)
df_train, df_val = train_test_split(df_target, test_size=0.2, stratify=df_target['label'], random_state=42)

train_loader = DataLoader(FCDataset(df_train), batch_size=16, shuffle=True)
val_loader = DataLoader(FCDataset(df_val), batch_size=16, shuffle=False)

model = GraphAutoencoder(num_nodes=100, input_dim=100, hidden_dim=64, embedding_dim=128)
model.encoder.load_state_dict(torch.load(encoder_weights))
model.encoder.eval()
for param in model.encoder.parameters():
    param.requires_grad = False

index = faiss.read_index(index_path)
kb_embeddings = np.load(kb_embeddings_path).astype('float32')
K = 10

# Only change: AttentionMechanismLinear instead of AttentionMechanism
attention_model = AttentionMechanismLinear(embedding_dim=128)
rac_model = RAC_Model(model.encoder, attention_model, embedding_dim=128)

optimizer = torch.optim.Adam(rac_model.parameters(), lr=0.001)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=3)
loss_fn = torch.nn.BCELoss()

num_epochs = 200
all_val_accuracies = []

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
    
    rac_model.eval()
    val_loss = 0
    correct = 0
    total = 0
    
    with torch.no_grad():
        for val_data in val_loader:
            val_query = model.encoder(val_data.x, val_data.edge_index, val_data.edge_weight, val_data.batch)
            val_query_np = val_query.cpu().numpy().astype('float32')
            _, val_neighbor_indices = index.search(val_query_np, K)
            val_retrieved = torch.from_numpy(kb_embeddings[val_neighbor_indices]).to(val_query.device)
            
            val_predictions, _ = rac_model(val_data.x, val_data.edge_index, val_data.edge_weight, val_data.batch, val_retrieved)
            val_loss += loss_fn(val_predictions.squeeze(), val_data.y.float().squeeze()).item()
            predicted_labels = (val_predictions.squeeze() > 0.5).float()
            true_labels = val_data.y.float().squeeze()
            
            if true_labels.dim() == 0:
                true_labels = true_labels.unsqueeze(0)
                predicted_labels = predicted_labels.unsqueeze(0)
                
            correct += (predicted_labels == true_labels).sum().item()
            total += val_data.y.size(0)
    
    avg_val_loss = val_loss / len(val_loader)
    val_accuracy = 100 * correct / total
    all_val_accuracies.append(val_accuracy)

    scheduler.step(avg_val_loss)
    
    if (epoch + 1) % 10 == 0:
        print(f"Epoch {epoch+1:02d}/{num_epochs} | Train Loss: {total_loss/len(train_loader):.4f} | Val Loss: {avg_val_loss:.4f} | Val Acc: {val_accuracy:.2f}%")

torch.save(rac_model.state_dict(), os.path.join(output_dir, 'rac_model_linear.pth'))
print("RAC Model (Linear Attention) Training Complete and Saved.")
mean_val_acc = sum(all_val_accuracies) / len(all_val_accuracies)
print(f"Mean Validation Accuracy (across all {num_epochs} epochs): {mean_val_acc:.2f}%")
print(f"Final Epoch Validation Accuracy: {all_val_accuracies[-1]:.2f}%")