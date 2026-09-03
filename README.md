# practical-project
## A Retrieval-Augmented Framework for Neurological Classification
# Practical Project

## Retrieval-Augmented Graph Neural Network for Parkinson's Disease Classification

This repository implements a **retrieval-augmented classification (RAC) framework** for classifying Parkinson's disease (PD) from resting-state functional connectivity (FC) matrices.

The pipeline combines:

* Graph Neural Networks (GNNs)
* Graph Autoencoder (GAE) representation learning
* FAISS nearest-neighbor retrieval
* Attention-based retrieval augmentation
* A GNN baseline for comparison

The current experiments use **PPMI** and **Neurocon** as an external knowledge base and **TaoWu** as the target classification dataset.

---

## Pipeline Overview

```text
Raw FC matrices
      │
      ▼
01_Prepare_Data.py
      │
      ▼
Graph-ready FC data
      │
      ▼
02_Train_GAE.py
      │
      ▼
Pretrained GAE encoder
      │
      ▼
03_Build_index.py
      │
      ▼
FAISS knowledge base
      │
      ├───────────────┐
      ▼               ▼
RAC models        Baseline GNN
      │               │
      └───────┬───────┘
              ▼
     06_Evaluate_models.py
```

---

# Repository Structure

```text
practical-project/
│
├── 01_Prepare_Data.py
├── 02_Train_GAE.py
├── 03_Build_index.py
│
├── 04_Train_rac_model.py
├── 04_final1.py
├── 04_final2.py
│
├── 05_Train_baseline_model.py
├── 05_final.py
│
├── 06_Evaluate_models.py
│
├── models.py
└── utils.py
```

---

# Python Files

## `01_Prepare_Data.py`

Prepares the functional-connectivity data for all later experiments.

The script:

* reads metadata for **Neurocon, PPMI, and TaoWu**,
* removes Prodromal and SWEDD subjects,
* searches each subject folder for the Schaefer-100 correlation matrix,
* loads the `.mat` FC matrix,
* converts it to NumPy `.npy` format,
* converts diagnosis to a binary label:

  * `1 = PD`
  * `0 = Control`
* creates a common metadata table.

Main output:

```text
master_metadata.csv
```

with fields such as:

```text
subject_id
label
diagnosis
dataset_source
npy_path
```

---

## `utils.py`

Contains the `FCDataset` class used by PyTorch Geometric.

Each 100 × 100 FC matrix is converted into a graph with:

* **100 nodes** representing brain regions,
* identity matrices as node features,
* a fully connected graph structure,
* FC values as edge weights.

It also stores the original FC matrix as the reconstruction target for the graph autoencoder.

The file additionally provides a helper function for loading FAISS indices.

---

## `models.py`

Contains all neural-network architectures used in the project.

### `GAEEncoder`

Encodes an FC graph into a **128-dimensional graph-level embedding** using:

```text
GCNConv
→ ReLU
→ Dropout
→ GCNConv
→ ReLU
→ Dropout
→ Linear
→ Global Mean Pooling
```

### `GraphAutoencoder`

Combines the GAE encoder with a decoder that reconstructs the original 100 × 100 FC matrix.

The learned 128-D bottleneck representation is later used for retrieval.

### `AttentionMechanism`

Uses dot-product attention between the target subject embedding and retrieved knowledge-base embeddings.

### `AttentionMechanismLinear`

A learnable attention variant that projects the query and retrieved embeddings before computing attention scores.

### `RAC_Model`

The Retrieval-Augmented Classifier combines:

```text
128-D target embedding
+
128-D retrieved context
=
256-D augmented representation
```

The representation is passed to an MLP to predict PD versus control.

### `Baseline_GNN`

Uses the same type of graph encoder but **without retrieval or external knowledge**.

It is trained from scratch on the TaoWu dataset.

---

## `02_Train_GAE.py`

Trains the graph autoencoder using the external knowledge-base datasets:

```text
PPMI + Neurocon
```

The GAE learns to reconstruct the original FC matrices from their graph embeddings.

Main configuration:

```text
Nodes:          100
Hidden size:    64
Embedding size: 128
Batch size:     32
Epochs:         100
Learning rate:  0.0001
Loss:           MSE
```

Only the trained encoder is saved:

```text
gae_encoder.pth
```

This encoder is reused during retrieval.

---

## `03_Build_index.py`

Uses the pretrained GAE encoder to generate embeddings for all PPMI and Neurocon subjects.

The embeddings are stored in a FAISS nearest-neighbor index using:

```python
faiss.IndexFlatL2(128)
```

which performs exact Euclidean-distance search.

Outputs:

```text
knowledge_base.index
kb_embeddings.npy
kb_metadata_indexed.csv
```

These files connect each FAISS vector to its corresponding knowledge-base subject.

---

## `04_Train_rac_model.py`

Trains the original Retrieval-Augmented Classifier on **TaoWu** using an 80/20 stratified train-validation split.

For every TaoWu subject:

```text
FC graph
   ↓
Pretrained GAE encoder
   ↓
128-D query embedding
   ↓
FAISS search
   ↓
10 nearest PPMI/Neurocon subjects
   ↓
Attention
   ↓
Retrieved context
   ↓
RAC classifier
```

The pretrained GAE encoder remains frozen during RAC training.

Main settings:

```text
K neighbors:    10
Batch size:     16
Epochs:         200
Learning rate:  0.001
Loss:           Binary Cross Entropy
```

Output:

```text
rac_model.pth
```

---

## `04_final1.py`

A **5-fold cross-validation** version of the RAC model.

It uses the original dot-product `AttentionMechanism`.

The RAC classifier is reinitialized for each fold to avoid information leakage.

Outputs:

```text
rac_model_fold_1.pth
...
rac_model_fold_5.pth
```

The script reports mean and final validation accuracy across the five folds.

---

## `04_final2.py`

A second 5-fold RAC experiment using:

```python
AttentionMechanismLinear
```

instead of direct dot-product attention.

The model learns query and key projections before computing attention scores.

Outputs:

```text
rac_model_linear_fold_1.pth
...
rac_model_linear_fold_5.pth
```

This experiment allows comparison between **fixed similarity-based attention** and **learnable attention**.

---

## `05_Train_baseline_model.py`

Trains the baseline classifier on TaoWu.

Unlike RAC, the baseline:

* does not use PPMI or Neurocon retrieval,
* does not use FAISS,
* does not use attention,
* starts with a randomly initialized graph encoder.

It therefore measures how well the target dataset can be classified without retrieval augmentation.

Main settings:

```text
Batch size:     16
Epochs:         100
Learning rate:  0.001
Loss:           Binary Cross Entropy
```

Output:

```text
baseline_model.pth
```

---

## `05_final.py`

The **5-fold cross-validation version** of the baseline model.

A completely new baseline model is initialized for each fold.

Outputs:

```text
baseline_model_fold_1.pth
...
baseline_model_fold_5.pth
```

Its performance can be compared with `04_final1.py` and `04_final2.py`.

---

## `06_Evaluate_models.py`

Evaluates three classifiers:

```text
RAC v1 — Dot-product attention
RAC v2 — Linear attention
Baseline GNN
```

For the RAC models, each TaoWu subject is first encoded and its 10 nearest knowledge-base embeddings are retrieved through FAISS.

The script reports:

* Accuracy
* ROC-AUC
* Confusion matrix

It also identifies subjects where a RAC model predicts correctly while the baseline predicts incorrectly.

Detailed predictions are saved to:

```text
test_evaluation_results.csv
```

---

# Model Comparison

| Model        | Pretrained Encoder | External Retrieval | Attention        |
| ------------ | ------------------ | ------------------ | ---------------- |
| Baseline GNN | No                 | No                 | No               |
| RAC v1       | Yes                | Yes                | Dot-product      |
| RAC v2       | Yes                | Yes                | Learnable linear |

The main research question is whether adding information retrieved from external datasets improves classification compared with learning only from the target dataset.

---

# Installation

Main dependencies:

```bash
pip install numpy pandas scipy scikit-learn torch torch-geometric faiss-cpu
```

---

# Running the Pipeline

First prepare the data and build the knowledge base:

```bash
python 01_Prepare_Data.py
python 02_Train_GAE.py
python 03_Build_index.py
```

For the 5-fold RAC experiments:

```bash
python 04_final1.py
python 04_final2.py
```

For the 5-fold baseline:

```bash
python 05_final.py
```

The original hold-out experiments are available in:

```text
04_Train_rac_model.py
05_Train_baseline_model.py
```

---

# Important Note

The scripts currently contain local Windows paths such as:

```python
C:\Users\nikna\Documents\pp_datasets\...
```

These paths must be changed to match the local dataset and output directories before running the code.

The current evaluation script also expects single-model checkpoint filenames, while the `*_final.py` experiments produce separate checkpoints for each cross-validation fold. The evaluation procedure should therefore be adjusted depending on whether the hold-out or 5-fold experimental setup is being used.

---

## Summary

The project investigates whether **retrieval from external functional-connectivity datasets can improve neurological classification**.

The main workflow is:

```text
FC Matrix
   ↓
Graph Neural Network
   ↓
Graph Embedding
   ↓
FAISS Retrieval
   ↓
Attention over Similar Subjects
   ↓
PD / Control Classification
```

The RAC models are compared against a graph neural network trained from scratch without retrieval to quantify the contribution of external knowledge.
