import os
import torch
import torch.nn.functional as F
import numpy as np
from torch_geometric.data import Data, DataLoader
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold, cross_validate
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

# --- 1. IMPORT YOUR MODEL ---
from models import GraphAutoencoder 

# Explicitly use CPU
device = torch.device('cpu')
print("Running on CPU mode.")
# data_path = r'C:\Users\nikna\Documents\pp_datasets\processed_data_1'

# --- 2. DATA LOADING & GRAPH CONVERSION ---
def load_gae_data(data_path):
    files = [f for f in os.listdir(data_path) if f.endswith('.npy')]
    data_list = []
    
    for f in files:
        matrix = np.load(os.path.join(data_path, f))
        
        fname = f.lower()
        if 'patient' in fname:
            label = 1
        elif 'control' in fname:
            label = 0
        else:
            continue

        # Extract edges from FC matrix (non-zero entries)
        edge_indices = np.array(np.nonzero(matrix))
        edge_index = torch.tensor(edge_indices, dtype=torch.long)
        edge_attr = torch.tensor(matrix[np.nonzero(matrix)], dtype=torch.float)
        
        # Node Features: 100x100 Identity Matrix
        x = torch.eye(100, dtype=torch.float) 
        
        y_tensor = torch.tensor([label], dtype=torch.long)
        orig_matrix = torch.tensor(matrix, dtype=torch.float)
        
        # Mapping 'edge_attr' to 'edge_weight' as per your GAE class definition
        d = Data(x=x, edge_index=edge_index, edge_weight=edge_attr, y=y_tensor, orig=orig_matrix)
        data_list.append(d)
        
    return data_list

# --- 3. TRAINING THE GAE ---
DATA_PATH =  r'C:\Users\nikna\Documents\pp_datasets\processed_data_1' # <--- Update this to your local path
dataset = load_gae_data(DATA_PATH)

# Using num_workers=0 is usually more stable for CPU-only training in Windows/macOS
loader = DataLoader(dataset, batch_size=16, shuffle=True, num_workers=0)

# Initialize model (adjust hidden_dim/embedding_dim if needed)
model = GraphAutoencoder(num_nodes=100, input_dim=100, hidden_dim=64, embedding_dim=128).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

print(f"Starting CPU training for {len(dataset)} subjects...")
model.train()
for epoch in range(100):
    total_loss = 0
    for batch in loader:
        optimizer.zero_grad()
        
        # Forward pass using CPU
        recon = model(batch.x, batch.edge_index, batch.edge_weight, batch.batch)
        
        # Compare reconstructed matrix to original input
        target = batch.orig.view(-1, 100, 100)
        loss = F.mse_loss(recon, target)
        
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    
    if (epoch+1) % 20 == 0:
        print(f"Epoch {epoch+1} | MSE Loss: {total_loss/len(loader):.4f}")

# --- 4. FEATURE EXTRACTION ---
print("\nExtracting 128-dimensional latent embeddings...")
model.eval()
X_latent, y_labels = [], []

with torch.no_grad():
    # Extract one by one to keep the mapping clean
    full_loader = DataLoader(dataset, batch_size=1, shuffle=False)
    for batch in full_loader:
        z = model.encode(batch.x, batch.edge_index, batch.edge_weight, batch.batch)
        X_latent.append(z.numpy())
        y_labels.append(batch.y.item())

X_latent = np.concatenate(X_latent, axis=0)
y_labels = np.array(y_labels)

# Save the reduced features for later use
np.save('pd_hc_embeddings.npy', X_latent)
np.save('pd_hc_labels.npy', y_labels)

# --- 5. RANDOM FOREST EVALUATION ---
print("\nRunning Stratified 5-Fold Cross-Validation...")
rf_pipeline = Pipeline([
    ('scaler', StandardScaler()), 
    ('rf', RandomForestClassifier(n_estimators=1000, max_features='sqrt', class_weight='balanced', random_state=42))
])

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
scores = cross_validate(rf_pipeline, X_latent, y_labels, cv=cv, scoring=['accuracy', 'roc_auc', 'recall'])

print("\n" + "="*30)
print(f"FINAL CPU RESULTS:")
print(f"Accuracy: {scores['test_accuracy'].mean():.2f} (±{scores['test_accuracy'].std():.2f})")
print(f"AUC-ROC:  {scores['test_roc_auc'].mean():.2f}")
print(f"Recall:   {scores['test_recall'].mean():.2f}")
print("="*30)

import numpy as np

def check_array_health(arr, name="Array"):
    has_nan = np.isnan(arr).any()
    has_inf = np.isinf(arr).any()
    
    print(f"--- Health Check: {name} ---")
    print(f"Contains NaN: {has_nan}")
    print(f"Contains Inf: {has_inf}")
    if has_nan or has_inf:
        print(f"Total problematic cells: {np.isnan(arr).sum() + np.isinf(arr).sum()}")
    print("-" * 25)

# Usage:
check_array_health(X_latent, "GAE Embeddings")