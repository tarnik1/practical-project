# SCRIPT 02: TRAIN GRAPH AUTOENCODER
# GOAL: Train the GAE on the knowledge base (PPMI, Neurocon).

# 1. Import libraries
#    (e.g., torch, pandas, and from models import GraphAutoencoder, from utils import FCDataset)

# 2. Load the master metadata.csv
#    - (e.g., df = pd.read_csv(OUTPUT_DIR / 'metadata.csv'))

# 3. Filter the metadata for the knowledge base
#    - (e.g., df_kb = df[df['dataset_source'].isin(['PPMI', 'Neurocon'])])

# 4. Create PyTorch Dataset and DataLoader
#    - kb_dataset = FCDataset(df_kb)
#    - kb_dataloader = DataLoader(kb_dataset, batch_size=32, shuffle=True)

# 5. Initialize the GAE model, optimizer, and loss function
#    - model = GraphAutoencoder(...)
#    - optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
#    - loss_fn = torch.nn.MSELoss()  # (Reconstruction loss)

# 6. Start the training loop
#    - for epoch in range(num_epochs):
#    -   for (data, label) in kb_dataloader:
#    -     # ... 1. Zero gradients (optimizer.zero_grad())
#    -     # ... 2. Get model output (reconstructed_matrix = model(data))
#    -     # ... 3. Calculate loss (loss = loss_fn(reconstructed_matrix, original_matrix))
#    -     # ... 4. Backpropagate (loss.backward())
#    -     # ... 5. Update weights (optimizer.step())
#    -   # ... Print epoch loss

# 7. Save the trained *encoder* part of the GAE
#    - (e.g., torch.save(model.encoder.state_dict(), 'gae_encoder.pth'))