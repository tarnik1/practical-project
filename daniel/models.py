# HELPER SCRIPT: MODEL ARCHITECTURES
# GOAL: Define all neural network architectures in one place.

# 1. Import necessary libraries
#    (e.g., torch, torch.nn, torch_geometric.nn)

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, global_mean_pool

# 2. Define the Graph Autoencoder (GAE)
#    - class GraphAutoencoder(torch.nn.Module):
#    -   def __init__(self, num_nodes, embedding_dim):
#    -     # ... define encoder layers (e.g., GCNConv)
#    -     # ... define readout/pooling layer (e.g., global_mean_pool)
#    -     # ... define decoder layers
#    -   def encode(self, x, edge_index, edge_weight):
#    -     # ... pass data through GCN layers
#    -     # ... apply pooling to get one graph-level vector
#    -     # ... return graph_level_vector
#    -   def decode(self, graph_level_vector):
#    -     # ... try to reconstruct the FC matrix
#    -     # ... return reconstructed_matrix
#    -   def forward(self, x, edge_index, edge_weight):
#    -     # ... call encode()
#    -     # ... call decode()
#    -     # ... return reconstructed_matrix
# models.py

class GAEEncoder(nn.Module):
    def __init__(self, input_dim, hidden_dim, embedding_dim):
        super(GAEEncoder, self).__init__()
        self.conv1 = GCNConv(input_dim, hidden_dim, normalize=False, add_self_loops=False)
        self.conv2 = GCNConv(hidden_dim, hidden_dim, normalize=False, add_self_loops=False)
        self.lin_encode = nn.Linear(hidden_dim, embedding_dim)

    def forward(self, x, edge_index, edge_weight, batch=None):
        """Encoder using absolute edge weights for stability"""
        edge_weight = torch.abs(edge_weight)
        
        x = self.conv1(x, edge_index, edge_weight=edge_weight)
        x = F.relu(x)
        x = F.dropout(x, p=0.3, training=self.training)
        
        x = self.conv2(x, edge_index, edge_weight=edge_weight)
        x = F.relu(x)
        x = F.dropout(x, p=0.3, training=self.training)

        x = self.lin_encode(x)

        if batch is None:
            batch = torch.zeros(x.size(0), dtype=torch.long, device=x.device)
            
        graph_level_vector = global_mean_pool(x, batch)
        return graph_level_vector

class GraphAutoencoder(torch.nn.Module):
    def __init__(self, num_nodes, input_dim, hidden_dim, embedding_dim):
        super(GraphAutoencoder, self).__init__()
        self.num_nodes = num_nodes
        self.embedding_dim = embedding_dim

        # ENCODER
        self.encoder = GAEEncoder(input_dim, hidden_dim, embedding_dim)

        # DECODER
        self.decoder = nn.Sequential(
            nn.Linear(embedding_dim, hidden_dim * 2),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim * 2, num_nodes * num_nodes)
        )

    def decode(self, graph_level_vector):
        reconstruction_flat = self.decoder(graph_level_vector)
        batch_size = graph_level_vector.size(0)
        reconstruction_matrix = reconstruction_flat.view(batch_size, self.num_nodes, self.num_nodes)
        return reconstruction_matrix

    def forward(self, x, edge_index, edge_weight, batch=None):
        z = self.encoder(x, edge_index, edge_weight, batch)
        out = self.decode(z)
        return out

# 3. Define the Attention Mechanism
#    - class AttentionMechanism(torch.nn.Module):
#    -   def __init__(self, embedding_dim):
#    -     # ... define layers to calculate scores (e.g., nn.Linear)
#    -   def forward(self, v_query, V_retrieved):
#    -     # ... calculate dot-product or additive attention scores
#    -     # ... apply softmax to get weights
#    -     # ... compute weighted average of V_retrieved
#    -     # ... return v_context

class AttentionMechanism(nn.Module):
    def __init__(self, embedding_dim):
        super(AttentionMechanism, self).__init__()
        # In this 'Pure Dot-Product' version, we don't need linear layers here
        # because we are comparing the v_query and V_retrieved directly.
        pass

    def forward(self, v_query, V_retrieved):
        """
        v_query: (Batch, 128) 
        V_retrieved: (Batch, k, 128) - e.g., (Batch, 5, 128)
        """
        # STEP 1: Calculate Relevance Scores (Dot-Product)
        # We need to make v_query (Batch, 128, 1) to multiply with V_retrieved (Batch, 5, 128)
        # This calculates: score_i = v_query · v_i for each of the k neighbors
        query = v_query.unsqueeze(2) 
        scores = torch.bmm(V_retrieved, query) # Result shape: (Batch, k, 1)
        
        # STEP 2: Normalize Scores to Weights (Softmax)
        # This turns raw scores into probabilities (e.g., [0.05, 0.90, 0.05])
        weights = F.softmax(scores, dim=1) # Result shape: (Batch, k, 1)
        
        # STEP 3: Create the Context Vector (Weighted Average)
        # Multiply each neighbor vector by its corresponding weight and sum them up
        # weights: (Batch, k, 1), V_retrieved: (Batch, k, 128)
        v_context = torch.sum(weights * V_retrieved, dim=1) # Result shape: (Batch, 128)
        
        return v_context, weights # Returning weights for future explainability!

# 4. Define the full Retrieval-Augmented Classifier (RAC)
#    - class RAC_Model(torch.nn.Module):
#    -   def __init__(self, gae_encoder, attention_model, embedding_dim):
#    -     # ... store the encoder and attention models
#    -     # ... define the final classification head (e.g., nn.Linear)
#    -   def forward(self, x, edge_index, edge_weight, V_retrieved):
#    -     # ... 1. v_query = self.gae_encoder(x, edge_index, edge_weight)
#    -     # ... 2. v_context = self.attention_model(v_query, V_retrieved)
#    -     # ... 3. v_augmented = torch.cat((v_query, v_context), dim=1)
#    -     # ... 4. prediction = self.classification_head(v_augmented)
#    -     # ... return prediction

class RAC_Model(nn.Module):
    def __init__(self, gae_encoder, attention_model, embedding_dim):
        super(RAC_Model, self).__init__()
        self.gae_encoder = gae_encoder
        self.attention_model = attention_model
        
        # Final classification head: Input is 256 (128 original + 128 context)
        self.classification_head = nn.Sequential(
            nn.Linear(embedding_dim * 2, 64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, 1),
            nn.Sigmoid()
        )
        
    def forward(self, x, edge_index, edge_weight, batch, V_retrieved):
        # STEP 1 (from Plan): Query Encoding
        v_query = self.gae_encoder(x, edge_index, edge_weight, batch)
        
        # STEP 2 & 3 (from Plan): Knowledge Retrieval & Contextual Augmentation
        # (Note: V_retrieved is passed from the script that queries FAISS)
        v_context, attn_weights = self.attention_model(v_query, V_retrieved)
        
        # STEP 4 (from Plan): Final Augmentation (Concatenation)
        v_augmented = torch.cat((v_query, v_context), dim=1) # Result: (Batch, 256)
        
        # Final Prediction
        prediction = self.classification_head(v_augmented)
        
        return prediction, attn_weights

# 5. Define the Baseline GNN Classifier
#    - class Baseline_GNN(torch.nn.Module):
#    -   def __init__(self, gae_encoder, embedding_dim):
#    -     # ... store the encoder
#    -     # ... define a classification head (same as RAC's)
#    -   def forward(self, x, edge_index, edge_weight):
#    -     # ... 1. v_embedding = self.gae_encoder(x, edge_index, edge_weight)
#    -     # ... 2. prediction = self.classification_head(v_embedding)
#    -     # ... return prediction

class Baseline_GNN(nn.Module):
    def __init__(self, gae_encoder, embedding_dim):
        super(Baseline_GNN, self).__init__()
        # Store the pre-trained encoder (same one used in RAC)
        self.gae_encoder = gae_encoder
        
        # Define a classification head
        # NOTE: This takes embedding_dim (128) as input, 
        # whereas RAC took embedding_dim * 2 (256).
        self.classification_head = nn.Sequential(
            nn.Linear(embedding_dim, 64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, 1),
            nn.Sigmoid()
        )
        
    def forward(self, x, edge_index, edge_weight, batch):
        # 1. Generate embedding from the target brain (No retrieval step!)
        v_embedding = self.gae_encoder(x, edge_index, edge_weight, batch)
        
        # 2. Make prediction based ONLY on this subject's data
        prediction = self.classification_head(v_embedding)
        
        return prediction