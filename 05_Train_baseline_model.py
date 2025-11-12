# SCRIPT 05: TRAIN THE BASELINE (AB-INITIO) MODEL
# GOAL: Train a simple GNN on the TaoWu dataset from scratch.

# 1. Import libraries
#    (e.g., torch, pandas)
#    (from models import GraphAutoencoder, Baseline_GNN)
#    (from utils import FCDataset)

# 2. Load metadata, create train/val splits for TaoWu
#    - (Same as step 2 & 3 in script 04)

# 3. Initialize the Baseline model
#    - # Note: We initialize a NEW encoder, not the pre-trained one
#    - gae_encoder_scratch = GraphAutoencoder(...).encoder
#    - baseline_model = Baseline_GNN(gae_encoder_scratch, ...)
#    - optimizer = torch.optim.Adam(baseline_model.parameters(), lr=0.001)
#    - loss_fn = torch.nn.BCEWithLogitsLoss()

# 4. Start the training loop
#    - for epoch in range(num_epochs):
#    -   for (data, label) in train_dataloader:
#    -     # ... 1. Get prediction (prediction = baseline_model(data))
#    -     # ... 2. Calculate loss
#    -     # ... 3. Backpropagate and update
#    -   # ... Run validation loop

# 5. Save the trained baseline model
#    - (e.g., torch.save(baseline_model.state_dict(), 'baseline_model.pth'))