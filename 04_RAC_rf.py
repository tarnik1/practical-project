# SCRIPT 04_RF: RANDOM FOREST RETRIEVAL-AUGMENTED CLASSIFIER (5-Fold CV)
# GOAL: Test if a Classical ML algorithm can solve the frozen FAISS space.

import os
import torch
import pandas as pd
import numpy as np
import faiss
from torch_geometric.loader import DataLoader
from sklearn.model_selection import StratifiedKFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, roc_auc_score
from models import GraphAutoencoder
from utils import FCDataset

output_dir = r'C:\Users\nikna\Documents\pp_datasets\processed_data_1'
master_csv = os.path.join(output_dir, "master_metadata.csv")
encoder_weights = os.path.join(output_dir, 'gae_encoder.pth')
index_path = os.path.join(output_dir, 'knowledge_base.index') 
kb_embeddings_path = os.path.join(output_dir, 'kb_embeddings.npy')

# 1. Load Metadata
df = pd.read_csv(master_csv)
df_target = df[df['dataset_source'] == 'TaoWu'].reset_index(drop=True)

# 2. Load Frozen GAE & FAISS 
base_model = GraphAutoencoder(num_nodes=100, input_dim=100, hidden_dim=64, embedding_dim=128)
base_model.encoder.load_state_dict(torch.load(encoder_weights), strict=False)
gae_encoder = base_model.encoder
gae_encoder.eval()

for param in gae_encoder.parameters():
    param.requires_grad = False

index = faiss.read_index(index_path)
kb_embeddings = np.load(kb_embeddings_path).astype('float32')
K = 5

# 3. Helper Function to Build the 256-D Dataset
def extract_rf_features(dataloader):
    X_features = []
    y_labels = []
    
    with torch.no_grad():
        for data in dataloader:
            # clean_edge_weight = torch.nan_to_num(data.edge_weight, nan=0.0, posinf=1.0, neginf=-1.0)
            
            v_query = gae_encoder(data.x, data.edge_index, data.edge_weight, data.batch)
            v_query_np = v_query.cpu().numpy().astype('float32')
            
            _, neighbor_indices = index.search(v_query_np, K)
            v_retrieved_np = kb_embeddings[neighbor_indices] 
            v_context_np = np.mean(v_retrieved_np, axis=1) 
            
            v_augmented_np = np.concatenate((v_query_np, v_context_np), axis=1) 
            
            X_features.append(v_augmented_np)
            y_labels.append(data.y.cpu().numpy())
            
    return np.vstack(X_features), np.concatenate(y_labels)

# 4. 5-Fold Setup
n_splits = 5
skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
fold_acc_results = []
fold_auc_results = [] # <-- Track AUC separately

print(f"Starting {n_splits}-Fold Cross Validation with Random Forest...")

for fold, (train_idx, val_idx) in enumerate(skf.split(df_target, df_target['label'])):
    print(f"\n--- FOLD {fold + 1}/{n_splits} ---")
    
    df_train = df_target.iloc[train_idx].reset_index(drop=True)
    df_val = df_target.iloc[val_idx].reset_index(drop=True)
    
    train_loader = DataLoader(FCDataset(df_train), batch_size=16, shuffle=False)
    val_loader = DataLoader(FCDataset(df_val), batch_size=16, shuffle=False)
    
    X_train, y_train = extract_rf_features(train_loader)
    X_val, y_val = extract_rf_features(val_loader)
    
    rf_classifier = RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42)
    rf_classifier.fit(X_train, y_train)
    
    # 1. ACCURACY CALCULATION (Hard Predictions)
    val_predictions = rf_classifier.predict(X_val)
    fold_acc = accuracy_score(y_val, val_predictions)
    fold_acc_results.append(fold_acc)
    
    # 2. AUC-ROC CALCULATION (Probability Estimates)
    # predict_proba returns [prob_class_0, prob_class_1]. We slice [:, 1] to get the probability of PD.
    val_probs = rf_classifier.predict_proba(X_val)[:, 1]
    fold_auc = roc_auc_score(y_val, val_probs)
    fold_auc_results.append(fold_auc)
    
    print(f"Fold {fold + 1} | Accuracy: {fold_acc * 100:.2f}% | AUC-ROC: {fold_auc:.4f}")

print(f"\n======================================")
print(f"OVERALL 5-FOLD ACCURACY: {np.mean(fold_acc_results) * 100:.2f}% ± {np.std(fold_acc_results) * 100:.2f}%")
print(f"OVERALL 5-FOLD AUC-ROC:  {np.mean(fold_auc_results):.4f} ± {np.std(fold_auc_results):.4f}")
print(f"======================================")