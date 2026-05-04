# SCRIPT 04: TRAIN THE RETRIEVAL-AUGMENTED CLASSIFIER
# GOAL: Train the full RAC model on the TaoWu dataset.

# 1. Import libraries
#    (e.g., torch, pandas, faiss)
#    (from models import GraphAutoencoder, AttentionMechanism, RAC_Model)
#    (from utils import FCDataset, load_faiss_index)

import os
import torch
import pandas as pd
import numpy as np
import faiss
from torch_geometric.loader import DataLoader
from sklearn.model_selection import train_test_split
from models import GraphAutoencoder, AttentionMechanism, RAC_Model
from utils import FCDataset

output_dir = r'C:\Users\nikna\Documents\pp_datasets\processed_data_1'
master_csv = os.path.join(output_dir, "master_metadata.csv")
encoder_weights = os.path.join(output_dir, 'gae_encoder.pth')
index_path = os.path.join(output_dir, 'knowledge_base.index') # spatial organization of the embeddings
kb_embeddings_path = os.path.join(output_dir, 'kb_embeddings.npy')
kb_metadata_path = os.path.join(output_dir, 'kb_metadata_indexed.csv')

# 2. Load the metadata and filter for the *target* dataset
#    - (e.g., df_target = df[df['dataset_source'] == 'TaoWu'])
#    - Split df_target into df_train and df_val (e.g., 80/20 split)

df = pd.read_csv(master_csv)
df_target = df[df['dataset_source'] == 'TaoWu'].reset_index(drop=True)
df_train, df_val = train_test_split(df_target, test_size=0.2, stratify=df_target['label'], random_state=42)

# 3. Create PyTorch Datasets and DataLoaders for train and validation
#    - train_dataset = FCDataset(df_train)
#    - train_dataloader = DataLoader(train_dataset, ...)

train_loader = DataLoader(FCDataset(df_train), batch_size=16, shuffle=True)
val_loader = DataLoader(FCDataset(df_val), batch_size=16, shuffle=False)

# 4. Load the pre-trained GAE encoder
#    - gae_encoder = GraphAutoencoder(...).encoder
#    - gae_encoder.load_state_dict(torch.load('gae_encoder.pth'))

model = GraphAutoencoder(num_nodes=100, input_dim=100, hidden_dim=64, embedding_dim=128)
model.encoder.load_state_dict(torch.load(encoder_weights))
model.encoder.eval()
# while the gae eencoder has already been trained and therefore we keep its weights frozen (.eval()),
# we still need to train the attention mechanism and the MLP.
for param in model.encoder.parameters():
    param.requires_grad = False

# 5. Load the FAISS index
#    - index = load_faiss_index('knowledge_base.index')
#    - K = 5 # (Number of neighbors to retrieve)

index = faiss.read_index(index_path)
kb_embeddings = np.load(kb_embeddings_path).astype('float32')
K = 10

# 6. Initialize the full RAC model
#    - attention_model = AttentionMechanism(...)
#    - rac_model = RAC_Model(gae_encoder, attention_model, ...)
#    - optimizer = torch.optim.Adam(rac_model.parameters(), lr=0.001)
#    - loss_fn = torch.nn.BCEWithLogitsLoss() # (For classification)

attention_model = AttentionMechanism(embedding_dim=128)
rac_model = RAC_Model(model.encoder, attention_model, embedding_dim=128)

optimizer = torch.optim.Adam(rac_model.parameters(), lr=0.001)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=3)
loss_fn = torch.nn.BCELoss()
# BCELoss penalizes a model when it guesses wrong in a yes/no (PD vs. HC) scenario.
# stands for binary cross entropy

# 7. Start the training loop
#    - for epoch in range(num_epochs):
#    -   for (data, label) in train_dataloader:
#    -     # ... 1. Get the query vector (with no gradient)
#    -     #      with torch.no_grad():
#    -     #        v_query = gae_encoder(data)
#    -     # ... 2. Retrieve neighbors from FAISS (this is a CPU/Numpy step)
#    -     #      distances, indices = index.search(v_query.cpu().numpy(), K)
#    -     # ... 3. Get the actual retrieved vectors (V_retrieved)
#    -     #      (This is complex: you need to map 'indices' back to your
#    -     #      'all_embeddings' from script 03.
#    -     #      It's best to save 'all_embeddings' as a .npy file)
#    -     # ... 4. Pass everything to the model
#    -     #      prediction = rac_model(data, V_retrieved)
#    -     # ... 5. Calculate loss (loss = loss_fn(prediction, label))
#    -     # ... 6. Backpropagate and update weights
#    -   # ... Run a similar loop on the validation set to check performance

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
        _, neighbor_indices = index.search(v_query_np, K) # do I actually need the distances?
        
        # neighbor_indices is (Batch, K), we use it to slice our .npy array
        v_retrieved = torch.from_numpy(kb_embeddings[neighbor_indices]).to(v_query.device)
        
        # Forward Pass # the underscore takes the place of the attention_weights
        predictions, _ = rac_model(data.x, data.edge_index, data.edge_weight, data.batch, v_retrieved)
        
        loss = loss_fn(predictions.squeeze(), data.y.float().squeeze())
        # BCELoss compares the model's prediction to the actual label and then calculates a penalty score.
        loss.backward()

        torch.nn.utils.clip_grad_norm_(rac_model.parameters(), max_norm=1.0)

        optimizer.step()
        # PyTorch calculates which neurons in the Attention layer and MLP caused the error (backward), 
        # and the optimizer slightly turns the dials to do better next time (step).
        
        total_loss += loss.item()
    
    rac_model.eval()
    val_loss = 0
    correct = 0
    total = 0
    
    with torch.no_grad(): # (No learning allowed)
        for val_data in val_loader:

            val_query = model.encoder(val_data.x, val_data.edge_index, val_data.edge_weight, val_data.batch)
            
            val_query_np = val_query.cpu().numpy().astype('float32')
            _, val_neighbor_indices = index.search(val_query_np, K)
            
            val_retrieved = torch.from_numpy(kb_embeddings[val_neighbor_indices]).to(val_query.device)
            
            val_predictions, _ = rac_model(val_data.x, val_data.edge_index, val_data.edge_weight, val_data.batch, val_retrieved)

            batch_loss = loss_fn(val_predictions.squeeze(), val_data.y.float().squeeze())
            val_loss += batch_loss.item()

            predicted_labels = (val_predictions.squeeze() > 0.5).float()
            true_labels = val_data.y.float().squeeze()

            if true_labels.dim() == 0:
                true_labels = true_labels.unsqueeze(0)
                predicted_labels = predicted_labels.unsqueeze(0)

            correct += (predicted_labels == true_labels).sum().item()
            total += true_labels.size(0)
    
    avg_val_loss = val_loss / len(val_loader)
    val_accuracy = 100 * correct / total
    
    all_val_accuracies.append(val_accuracy)
    
    # Update learning rate based on validation loss
    scheduler.step(avg_val_loss)
    
    if (epoch + 1) % 10 == 0:
        print(f"Epoch {epoch+1:02d}/{num_epochs} | Train Loss: {total_loss/len(train_loader):.4f} | Val Loss: {avg_val_loss:.4f} | Val Acc: {val_accuracy:.2f}%")    

# 8. Save the final trained RAC model
#    - (e.g., torch.save(rac_model.state_dict(), 'rac_model.pth'))

torch.save(rac_model.state_dict(), os.path.join(output_dir, 'rac_model.pth'))
print("RAC Model Training Complete and Saved.")
mean_val_acc = sum(all_val_accuracies) / len(all_val_accuracies)
print(f"Mean Validation Accuracy (across all {num_epochs} epochs): {mean_val_acc:.2f}%")
print(f"Final Epoch Validation Accuracy: {all_val_accuracies[-1]:.2f}%")