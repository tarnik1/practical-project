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
from models import GraphAutoencoder, AttentionMechanism, RAC_Model, Baseline_GNN
from utils import FCDataset

output_dir = r'C:\Users\nikna\Documents\pp_datasets\processed_data_1'
master_csv = os.path.join(output_dir, "master_metadata.csv")
kb_embeddings_path = os.path.join(output_dir, 'kb_embeddings.npy')
index_path = os.path.join(output_dir, 'knowledge_base.index')

# Weights from Scripts 02, 04, and 05
encoder_weights = os.path.join(output_dir, 'gae_encoder.pth')
rac_weights = os.path.join(output_dir, 'rac_model.pth')
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
base_gae.load_state_dict(torch.load(encoder_weights))
attention = AttentionMechanism(embedding_dim=128)
rac_model = RAC_Model(gae_encoder=base_gae, attention_model=attention, embedding_dim=128)
rac_model.load_state_dict(torch.load(rac_weights))
rac_model.eval()

# 4. Load the trained Baseline model
#    - (Initialize Baseline_GNN, load 'baseline_model.pth')
#    - baseline_model.eval()

fresh_gae = GraphAutoencoder(100, 100, 64, 128)
baseline_model = Baseline_GNN(gae_encoder=fresh_gae, embedding_dim=128)
baseline_model.load_state_dict(torch.load(baseline_weights))
baseline_model.eval()

# 5. Initialize lists to store results
#    - all_labels = []
#    - rac_predictions = []
#    - baseline_predictions = []

all_labels = []
rac_probs = []
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
        
        v_query = rac_model.gae_encoder.encode(data.x, data.edge_index, data.edge_weight, data.batch)
        distances, indices = index.search(v_query.cpu().numpy().astype('float32'), k=5)
        v_retrieved = torch.from_numpy(kb_embeddings[indices]).to(v_query.device)
        
        rac_pred, _ = rac_model(data.x, data.edge_index, data.edge_weight, data.batch, v_retrieved)
        
        bl_pred = baseline_model(data.x, data.edge_index, data.edge_weight, data.batch)
        
        all_labels.append(data.y.item())
        rac_probs.append(rac_pred.squeeze().item())
        baseline_probs.append(bl_pred.squeeze().item())

# 7. Post-process results (convert logits to probabilities/classes)
#    - ...

rac_classes = [1 if p > 0.5 else 0 for p in rac_probs]
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
    print(f"AUC-ROC:   {roc_auc_score(labels, probs):.4f}")
    print("Confusion Matrix:")
    print(confusion_matrix(labels, classes))

print_metrics("RAC MODEL", all_labels, rac_probs, rac_classes)
print_metrics("BASELINE MODEL", all_labels, baseline_probs, baseline_classes)

# 9. (Optional) Save test results to a file

results_df = pd.DataFrame({
    'subject_id': df_test['subject_id'].values,
    'ground_truth': all_labels,
    'rac_prob': rac_probs,
    'rac_predicted_class': rac_classes,
    'baseline_prob': baseline_probs,
    'baseline_predicted_class': baseline_classes
})

# Cases where RAC was right and Baseline was wrong
results_df['rac_advantage'] = (results_df['rac_predicted_class'] == results_df['ground_truth']) & \
                             (results_df['baseline_predicted_class'] != results_df['ground_truth'])

results_csv_path = os.path.join(output_dir, 'test_evaluation_results.csv')
results_df.to_csv(results_csv_path, index=False)

print(f"\nDetailed results saved to: {results_csv_path}")
print(f"Number of cases where RAC outperformed Baseline: {results_df['rac_advantage'].sum()}")