"""Frozen-encoder nearest-neighbor composition, TaoWu held-out folds.

Uses the same frozen GAE and FAISS index as E2/E3/E4. Writes measured
per-subject counts and a compact 5-row manuscript table. Run from the
repository directory where models.py and utils.py are importable.

Usage:
    python sana_08_neighbor_composition.py --processed-dir processed_data_1 --k 10 --seed 42
"""
import argparse
from pathlib import Path
import numpy as np
import pandas as pd
import torch
import faiss
from sklearn.model_selection import StratifiedKFold
from torch_geometric.loader import DataLoader
from models import GraphAutoencoder
from utils import FCDataset


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--processed-dir', type=Path, required=True)
    p.add_argument('--k', type=int, default=10)
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--out-dir', type=Path, default=None)
    args = p.parse_args()
    root = args.processed_dir.resolve()
    out = args.out_dir or root / 'sana_results' / 'neighbor_composition'
    out.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(root / 'master_metadata.csv')
    def local_path(v):
        v = str(v)
        candidate = Path(v)
        if candidate.is_file():
            return str(candidate)
        return str(root / v.replace('\\', '/').split('/')[-1])
    df['npy_path'] = df['npy_path'].map(local_path)
    target = df[df['dataset_source'].eq('TaoWu')].reset_index(drop=True)
    kb = pd.read_csv(root / 'kb_metadata_indexed.csv')
    if len(target) != 40:
        print(f'Warning: expected 40 TaoWu subjects; found {len(target)}')
    for path in target['npy_path']:
        if not Path(path).is_file():
            raise FileNotFoundError(f'Missing target FC matrix: {path}')
    idx = faiss.read_index(str(root / 'knowledge_base.index'))
    if len(kb) != idx.ntotal:
        raise ValueError(f'FAISS index has {idx.ntotal} vectors; metadata has {len(kb)} rows')
    if args.k > idx.ntotal or args.k <= 0:
        raise ValueError('K must be between 1 and number of KB records')
    if not kb['label'].isin([0, 1]).all():
        raise ValueError('KB labels must be binary: HC=0, PD=1')
    torch.manual_seed(args.seed)
    model = GraphAutoencoder(num_nodes=100, input_dim=100, hidden_dim=64, embedding_dim=128)
    model.encoder.load_state_dict(torch.load(root / 'gae_encoder.pth', map_location='cpu', weights_only=True))
    model.encoder.eval()
    splits = StratifiedKFold(n_splits=5, shuffle=True, random_state=args.seed)
    rows = []
    for fold, (_, holdout) in enumerate(splits.split(target, target['label']), 1):
        subset = target.iloc[holdout].reset_index(drop=True)
        loader = DataLoader(FCDataset(subset), batch_size=1, shuffle=False)
        with torch.no_grad():
            for i, data in enumerate(loader):
                z = model.encoder(data.x, data.edge_index, data.edge_weight, data.batch)
                distances, neighbors = idx.search(z.cpu().numpy().astype('float32'), args.k)
                neighbors = neighbors[0]
                retrieved_labels = kb.iloc[neighbors]['label'].astype(int).to_numpy()
                n_pd = int(retrieved_labels.sum())
                n_hc = args.k - n_pd
                label = int(subset.iloc[i]['label'])
                subj = subset.iloc[i].get('subject_id', Path(subset.iloc[i]['npy_path']).stem)
                rows.append(dict(subject_id=str(subj), fold=fold, target_label=label,
                                 target_diagnosis='PD' if label else 'HC', K=args.k,
                                 pd_neighbors=n_pd, hc_neighbors=n_hc,
                                 pd_percent=100*n_pd/args.k, hc_percent=100*n_hc/args.k,
                                 same_diagnosis_percent=100*(n_pd if label else n_hc)/args.k,
                                 neighbor_index_ids=','.join(map(str,neighbors)),
                                 neighbor_distances=','.join(f'{v:.5g}' for v in distances[0])))
    per = pd.DataFrame(rows)
    if len(per) != len(target) or per['subject_id'].nunique() != len(target):
        print('Note: subject IDs may repeat; check uniqueness or filenames in metadata.')
    per.to_csv(out / 'neighbor_composition_per_subject.csv', index=False)
    # Programmatic example selection with unambiguous priorities, not prediction confidence.
    pd_cross = per[(per.target_label == 1) & (per.hc_neighbors > per.pd_neighbors)]
    hc_cross = per[(per.target_label == 0) & (per.pd_neighbors > per.hc_neighbors)]
    correct = per[per.same_diagnosis_percent > 50]
    selected = []
    for candidates in (pd_cross, hc_cross, correct):
        remaining = candidates[~candidates.subject_id.isin([r['subject_id'] for r in selected])]
        if remaining.empty:
            print('Warning: requested illustrative pattern absent; leaving this example out.')
        else:
            # Stable tie-breaking; no confidence-based selection.
            selected.append(remaining.sort_values(['fold','subject_id']).iloc[0].to_dict())
    table = []
    for j, row in enumerate(selected):
        table.append({'Target individual': f'Individual {chr(65+j)}', 'Fold': int(row['fold']),
                      'Target diagnosis': row['target_diagnosis'],
                      'PD neighbors, n (%)': f"{row['pd_neighbors']} ({row['pd_percent']:.0f}%)",
                      'HC neighbors, n (%)': f"{row['hc_neighbors']} ({row['hc_percent']:.0f}%)",
                      'Neighborhood pattern': 'Corresponding' if row['same_diagnosis_percent'] > 50 else 'Cross-diagnostic',
                      'subject_id (internal)': row['subject_id']})
    for label, diag in [(1,'PD'),(0,'HC')]:
        x = per[per.target_label == label]
        table.append({'Target individual': f'All {diag} target individuals', 'Fold': '—',
                      'Target diagnosis': diag,
                      'PD neighbors, n (%)': f"{x.pd_percent.mean():.1f} ± {x.pd_percent.std(ddof=1):.1f}%",
                      'HC neighbors, n (%)': f"{x.hc_percent.mean():.1f} ± {x.hc_percent.std(ddof=1):.1f}%",
                      'Neighborhood pattern': '—', 'subject_id (internal)': '—'})
    tbl = pd.DataFrame(table)
    tbl.to_csv(out / 'neighbor_composition_manuscript_table.csv', index=False)
    print(tbl.to_string(index=False))
    print(f'\nSaved to: {out}')
    print(f'KB PD fraction: {kb.label.mean()*100:.1f}% (imbalance reference; not a classifier baseline)')

if __name__ == '__main__':
    main()
