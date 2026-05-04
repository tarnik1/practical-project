#Prepare Data # SCRIPT 06: EVALUATE MODELS
# GOAL: Compare the RAC model vs. the Baseline on the test set.

# 1. Import libraries
#    (e.g., torch, pandas, sklearn.metrics)
#    (from models import ..., from utils import ...)

import os
import torch
import pandas as pd
import numpy as np
import faiss
from torch_geometric.loader import DataLoader
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, roc_auc_score, confusion_matrix, classification_report
from models import GraphAutoencoder, AttentionMechanism, AttentionMechanismLinear, RAC_Model, Baseline_GNN
from utils import FCDataset

output_dir = r'C:\Users\nikna\Documents\pp_datasets\processed_data_1'
master_csv = os.path.join(output_dir, "master_metadata.csv")
kb_embeddings_path = os.path.join(output_dir, 'kb_embeddings.npy')
index_path = os.path.join(output_dir, 'knowledge_base.index')

# Weights from Scripts 02, 04, and 05
encoder_weights = os.path.join(output_dir, 'gae_encoder.pth')
rac_v1_weights = os.path.join(output_dir, 'rac_model.pth')
rac_v2_weights = os.path.join(output_dir, 'rac_model_linear.pth')
baseline_weights = os.path.join(output_dir, 'baseline_model.pth')

# 2. Load the metadata and get the *test* split
#    - (e.g., df_target = df[df['dataset_source'] == 'TaoWu'])
#    - You need to have a held-out test set (e.g., from a CSV)
#    - test_dataset = FCDataset(df_test)
#    - test_dataloader = DataLoader(test_dataset, ...)

df = pd.read_csv(master_csv)
df_target = df[df['dataset_source'] == 'TaoWu'].reset_index(drop=True)

_, df_test = train_test_split(df_target, test_size=0.2, stratify=df_target['label'], random_state=42)
test_loader = DataLoader(FCDataset(df_test), batch_size=1, shuffle=False) # Batch 1 for individual analysis

# 3. Load the trained RAC model
#    - (Initialize RAC_Model, load 'rac_model.pth')
#    - (Also need to load GAE encoder and FAISS index for its forward pass)
#    - rac_model.eval()

index = faiss.read_index(index_path)
kb_embeddings = np.load(kb_embeddings_path).astype('float32')

base_gae = GraphAutoencoder(100, 100, 64, 128)
base_gae.encoder.load_state_dict(torch.load(encoder_weights))
base_gae.encoder.eval()

attention_v1 = AttentionMechanism(embedding_dim=128)
rac_model_v1 = RAC_Model(gae_encoder=base_gae.encoder, attention_model=attention_v1, embedding_dim=128)
rac_model_v1.load_state_dict(torch.load(rac_v1_weights))
rac_model_v1.eval()

attention_v2 = AttentionMechanismLinear(embedding_dim=128)
rac_model_v2 = RAC_Model(gae_encoder=base_gae.encoder, attention_model=attention_v2, embedding_dim=128)
rac_model_v2.load_state_dict(torch.load(rac_v2_weights))
rac_model_v2.eval()

# 4. Load the trained Baseline model
#    - (Initialize Baseline_GNN, load 'baseline_model.pth')
#    - baseline_model.eval()

fresh_gae = GraphAutoencoder(100, 100, 64, 128)
baseline_model = Baseline_GNN(gae_encoder=fresh_gae.encoder, embedding_dim=128)
baseline_model.load_state_dict(torch.load(baseline_weights))
baseline_model.eval()

# 5. Initialize lists to store results
#    - all_labels = []
#    - rac_predictions = []
#    - baseline_predictions = []

all_labels = []
rac_v1_probs = []
rac_v2_probs = []
baseline_probs = []

# 6. Start the evaluation loop
#    - with torch.no_grad():
#    -   for (data, label) in test_dataloader:
#    -     # ... 1. Get RAC prediction (requires retrieval step)
#    -     #      rac_pred = rac_model(...)
#    -     # ... 2. Get Baseline prediction
#    -     #      baseline_pred = baseline_model(data)
#    -     # ... 3. Store results
#    -     #      all_labels.append(label)
#    -     #      rac_predictions.append(rac_pred)
#    -     #      baseline_predictions.append(baseline_pred)

print("Starting Evaluation on TaoWu Test Set...")
with torch.no_grad():
    for data in test_loader:
        
        v_query = base_gae.encoder(data.x, data.edge_index, data.edge_weight, data.batch)
        distances, indices = index.search(v_query.cpu().numpy().astype('float32'), k=10)
        v_retrieved = torch.from_numpy(kb_embeddings[indices]).to(v_query.device)
        
        rac_v1_pred, _ = rac_model_v1(data.x, data.edge_index, data.edge_weight, data.batch, v_retrieved)
        rac_v2_pred, _ = rac_model_v2(data.x, data.edge_index, data.edge_weight, data.batch, v_retrieved)
        
        bl_pred = baseline_model(data.x, data.edge_index, data.edge_weight, data.batch)
        
        all_labels.append(data.y.item())
        rac_v1_probs.append(rac_v1_pred.squeeze().item())
        rac_v2_probs.append(rac_v2_pred.squeeze().item())
        baseline_probs.append(bl_pred.squeeze().item())

# 7. Post-process results (convert logits to probabilities/classes)
#    - ...

rac_v1_classes = [1 if p > 0.5 else 0 for p in rac_v1_probs]
rac_v2_classes = [1 if p > 0.5 else 0 for p in rac_v2_probs]
baseline_classes = [1 if p > 0.5 else 0 for p in baseline_probs]

# 8. Calculate and print metrics
#    - (e.g., from sklearn.metrics import accuracy_score, roc_auc_score)
#    - print("--- RAC Model Results ---")
#    - print(f"Accuracy: {accuracy_score(all_labels, rac_classes)}")
#    - print(f"AUC: {roc_auc_score(all_labels, rac_probs)}")
#    -
#    - print("--- Baseline Model Results ---")
#    - print(f"Accuracy: {accuracy_score(all_labels, baseline_classes)}")
#    - print(f"AUC: {roc_auc_score(all_labels, baseline_probs)}")

def print_metrics(name, labels, probs, classes):
    print(f"\n--- {name} Results ---")
    print(f"Accuracy:  {accuracy_score(labels, classes):.4f}")
    try:
        auc = roc_auc_score(labels, probs)
        print(f"AUC-ROC:   {auc:.4f}")
    except ValueError:
        print("AUC-ROC:   N/A (Only one class present in ground truth)")
    print("Confusion Matrix:")
    print(confusion_matrix(labels, classes))

print_metrics("RAC v1 (AVERAGE ATTENTION)", all_labels, rac_v1_probs, rac_v1_classes)
print_metrics("RAC v2 (LINEAR ATTENTION)", all_labels, rac_v2_probs, rac_v2_classes)
print_metrics("BASELINE MODEL (AB-INITIO)", all_labels, baseline_probs, baseline_classes)

# 9. (Optional) Save test results to a file

results_df = pd.DataFrame({
    'subject_id': df_test['subject_id'].values,
    'ground_truth': all_labels,
    'rac_v1_prob': rac_v1_probs,
    'rac_v1_predicted': rac_v1_classes,
    'rac_v2_prob': rac_v2_probs,
    'rac_v2_predicted': rac_v2_classes,
    'baseline_prob': baseline_probs,
    'baseline_predicted_class': baseline_classes
})

# Cases where RAC was right and Baseline was wrong
results_df['rac_v1_advantage'] = (results_df['rac_v1_predicted'] == results_df['ground_truth']) & \
                                 (results_df['baseline_predicted_class'] != results_df['ground_truth'])
results_df['rac_v2_advantage'] = (results_df['rac_v2_predicted'] == results_df['ground_truth']) & \
                                 (results_df['baseline_predicted_class'] != results_df['ground_truth'])

results_csv_path = os.path.join(output_dir, 'test_evaluation_results.csv')
results_df.to_csv(results_csv_path, index=False)

print(f"\nDetailed results saved to: {results_csv_path}")
print(f"Number of cases where RAC v1 outperformed Baseline: {results_df['rac_v1_advantage'].sum()}")
print(f"Number of cases where RAC v2 outperformed Baseline: {results_df['rac_v2_advantage'].sum()}")