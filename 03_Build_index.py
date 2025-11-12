# SCRIPT 03: BUILD FAISS INDEX
# GOAL: Create and save the FAISS index from the knowledge base embeddings.

# 1. Import libraries
#    (e.g., torch, pandas, faiss, numpy)
#    (from models import GraphAutoencoder, from utils import FCDataset)

# 2. Load the metadata and filter for the knowledge base (like in script 02).
#    - df_kb = ...

# 3. Create a DataLoader (no shuffling needed)
#    - kb_dataset = FCDataset(df_kb)
#    - kb_dataloader = DataLoader(kb_dataset, batch_size=32, shuffle=False)

# 4. Initialize the encoder and load its trained weights
#    - encoder = GraphAutoencoder(...).encoder
#    - encoder.load_state_dict(torch.load('gae_encoder.pth'))
#    - encoder.eval() # Set to evaluation mode

# 5. Loop through the dataloader and collect all embeddings
#    - all_embeddings = []
#    - with torch.no_grad():
#    -   for (data, label) in kb_dataloader:
#    -     # ... v_embedding = encoder(data)
#    -     # ... all_embeddings.append(v_embedding.cpu().numpy())
#    - all_embeddings = np.concatenate(all_embeddings, axis=0)

# 6. Create and populate the FAISS index
#    - EMBEDDING_DIM = 128
#    - index = faiss.IndexFlatL2(EMBEDDING_DIM) # (L2 distance = dot product for normalized vectors)
#    - index.add(all_embeddings)

# 7. Save the index to disk
#    - (e.g., faiss.write_index(index, 'knowledge_base.index'))