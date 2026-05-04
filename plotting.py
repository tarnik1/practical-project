# SCRIPT 07: PLOT COMBINED ROC CURVES
# GOAL: Visualize Sensitivity vs. Specificity for all three models.

import os
import torch
import pandas as pd
import numpy as np
import faiss
import matplotlib.pyplot as plt
from torch_geometric.loader import DataLoader
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_curve, auc

from models import GraphAutoencoder, AttentionMechanism, AttentionMechanismLinear, RAC_Model, Baseline_GNN
from utils import FCDataset

# --- 1. Define Paths ---
output_dir = r'C:\Users\nikna\Documents\pp_datasets\processed_data_1'
master_csv = os.path.join(output_dir, "master_metadata.csv")
kb_embeddings_path = os.path.join(output_dir, 'kb_embeddings.npy')
index_path = os.path.join(output_dir, 'knowledge_base.index')

encoder_weights = os.path.join(output_dir, 'gae_encoder.pth')
rac_v1_weights = os.path.join(output_dir, 'rac_model.pth')
rac_v2_weights = os.path.join(output_dir, 'rac_model_linear.pth')
baseline_weights = os.path.join(output_dir, 'baseline_model.pth')

# --- 2. Load Data ---
df = pd.read_csv(master_csv)
df_target = df[df['dataset_source'] == 'TaoWu'].reset_index(drop=True)
_, df_test = train_test_split(df_target, test_size=0.2, stratify=df_target['label'], random_state=42)
test_loader = DataLoader(FCDataset(df_test), batch_size=1, shuffle=False)

# --- 3. Load Models ---
index = faiss.read_index(index_path)
kb_embeddings = np.load(kb_embeddings_path).astype('float32')

base_gae = GraphAutoencoder(num_nodes=100, input_dim=100, hidden_dim=64, embedding_dim=128)
base_gae.encoder.load_state_dict(torch.load(encoder_weights))
base_gae.eval()

# RAC v1
rac_model_v1 = RAC_Model(gae_encoder=base_gae.encoder, attention_model=AttentionMechanism(128), embedding_dim=128)
rac_model_v1.load_state_dict(torch.load(rac_v1_weights))
rac_model_v1.eval()

# RAC v2
rac_model_v2 = RAC_Model(gae_encoder=base_gae.encoder, attention_model=AttentionMechanismLinear(128), embedding_dim=128)
rac_model_v2.load_state_dict(torch.load(rac_v2_weights))
rac_model_v2.eval()

# Baseline
baseline_model = Baseline_GNN(gae_encoder=GraphAutoencoder(100, 100, 64, 128).encoder, embedding_dim=128)
baseline_model.load_state_dict(torch.load(baseline_weights))
baseline_model.eval()

# --- 4. Get Predictions ---
print("Generating predictions for ROC Curve...")
all_labels, rac_v1_probs, rac_v2_probs, baseline_probs = [], [], [], []

with torch.no_grad():
    for data in test_loader:
        v_query = base_gae.encoder(data.x, data.edge_index, data.edge_weight, data.batch)
        distances, indices = index.search(v_query.cpu().numpy().astype('float32'), 10)
        v_retrieved = torch.from_numpy(kb_embeddings[indices]).to(v_query.device)
        
        rac_v1_pred, _ = rac_model_v1(data.x, data.edge_index, data.edge_weight, data.batch, v_retrieved)
        rac_v2_pred, _ = rac_model_v2(data.x, data.edge_index, data.edge_weight, data.batch, v_retrieved)
        bl_pred = baseline_model(data.x, data.edge_index, data.edge_weight, data.batch)
        
        all_labels.append(data.y.item())
        rac_v1_probs.append(rac_v1_pred.squeeze().item())
        rac_v2_probs.append(rac_v2_pred.squeeze().item())
        baseline_probs.append(bl_pred.squeeze().item())

# --- 5. Plot the ROC Curves ---
plt.figure(figsize=(10, 8))

# Calculate False Positive Rate (FPR) and True Positive Rate (TPR) for each
fpr_bl, tpr_bl, _ = roc_curve(all_labels, baseline_probs)
auc_bl = auc(fpr_bl, tpr_bl)

fpr_v1, tpr_v1, _ = roc_curve(all_labels, rac_v1_probs)
auc_v1 = auc(fpr_v1, tpr_v1)

fpr_v2, tpr_v2, _ = roc_curve(all_labels, rac_v2_probs)
auc_v2 = auc(fpr_v2, tpr_v2)

# Plot the lines
plt.plot(fpr_bl, tpr_bl, color='gray', linestyle='--', linewidth=3, label=f'Ab-Initio (AUC = {auc_bl:.3f})')
plt.plot(fpr_v1, tpr_v1, color='blue', linewidth=3.5, label=f'RAC v1 (AUC = {auc_v1:.3f})')
plt.plot(fpr_v2, tpr_v2, color='red', linewidth=3.5, label=f'RAC v2 (AUC = {auc_v2:.3f})')

# Plot the "Random Guess" diagonal line
plt.plot([0, 1], [0, 1], color='black', linestyle=':', label='Random Chance (AUC = 0.500)')

# Formatting
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate (1 - Specificity)', fontsize=14)
plt.ylabel('True Positive Rate (Sensitivity)', fontsize=14)
plt.title('Receiver Operating Characteristic (ROC)\nTarget Dataset: TaoWu', fontsize=16, fontweight='bold')
plt.legend(loc="lower right", fontsize=12)
plt.grid(alpha=0.3)

# Save and Show
plot_save_path = os.path.join(output_dir, 'combined_roc_curve.png')
plt.savefig(plot_save_path, dpi=300, bbox_inches='tight')
print(f"ROC Curve saved to: {plot_save_path}")

plt.show()