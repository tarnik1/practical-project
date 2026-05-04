import faiss
import numpy as np
import pandas as pd
import os

output_dir = r'C:\Users\nikna\Documents\pp_datasets\processed_data_1'
idx_path = os.path.join(output_dir, 'knowledge_base.index')
emb_path = os.path.join(output_dir, 'kb_embeddings.npy')
csv_path = os.path.join(output_dir, 'kb_metadata_indexed.csv')

print('Paths:')
print(' index:', idx_path)
print(' emb:  ', emb_path)
print(' csv:  ', csv_path)

try:
    emb = np.load(emb_path)
    print('\nEmbeddings: shape=', emb.shape, 'dtype=', emb.dtype)
except Exception as e:
    print('\nFailed to load embeddings:', e)
    emb = None

try:
    df = pd.read_csv(csv_path)
    print('Metadata: rows=', df.shape[0], 'cols=', df.shape[1])
    print('\nMetadata sample:\n', df.head(3).to_string())
except Exception as e:
    print('\nFailed to load metadata CSV:', e)
    df = None

try:
    idx = faiss.read_index(idx_path)
    print('\nFAISS index: ntotal=', idx.ntotal)
except Exception as e:
    print('\nFailed to read FAISS index:', e)
    idx = None

if emb is not None and df is not None and idx is not None:
    counts_equal = (emb.shape[0] == idx.ntotal == df.shape[0])
    print('\nCounts equal:', counts_equal)

    # Quick nearest-neighbour sanity check for first 3 vectors
    k = 5
    q = min(3, emb.shape[0])
    D, I = idx.search(emb[:q].astype('float32'), k)
    print('\nNN search (first', q, 'queries, k=', k, '):')
    print('Distances:\n', D)
    print('Indices:\n', I)
    print('\nMetadata for first query neighbors:')
    for col_idx in range(q):
        print('\nQuery', col_idx)
        print(df.iloc[I[col_idx]].to_string())

print('\nValidator finished')
