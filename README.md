# Retrieval-Augmented Graph Neural Networks for Parkinson's Disease Classification

This repository implements a **retrieval-augmented classification (RAC) framework** for Parkinson's disease (PD) classification using resting-state functional connectivity (FC) matrices.

The project investigates whether representations learned from external neuroimaging datasets and information retrieved from similar subjects can improve classification on a smaller target dataset.

The framework combines:

- Graph Neural Networks (GNNs)
- Graph Autoencoder (GAE) pretraining
- FAISS nearest-neighbor retrieval
- Retrieval-based feature aggregation
- Dot-product and learnable attention mechanisms
- Partial encoder fine-tuning
- Repeated stratified cross-validation
- Analysis of retrieved neighbors

The original project pipeline is preserved, alongside an extended ablation study comparing six model configurations (M0–M5).

---

## 1. Research Objective

The central research question is:

**Does retrieving information from an external functional-connectivity knowledge base improve Parkinson's disease classification beyond the benefits of pretrained graph representations?**

The study examines three components:

1. **Pretraining:** Whether a graph encoder pretrained on external functional-connectivity data improves classification.
2. **Retrieval augmentation:** Whether incorporating embeddings from similar external subjects provides additional predictive benefit.
3. **Retrieval quality:** Whether the diagnostic composition of retrieved neighbors differs between PD and healthy control (HC) subjects.

The extended experiments also examine the effect of retrieval-set size and partial fine-tuning of the pretrained encoder.

---

## 2. Datasets

Three resting-state functional-connectivity datasets are used.

| Dataset | Role | Subjects |
|---|---|---:|
| PPMI + Neurocon | External knowledge base | 169 |
| TaoWu | Target classification dataset | 40 |

### External Knowledge Base

The combined PPMI and Neurocon knowledge base contains:

- 139 PD subjects
- 30 healthy controls
- 169 subjects in total

The class distribution is approximately **82.2% PD** and **17.8% HC**.

These subjects are used to pretrain the graph encoder and construct the retrieval index.

### Target Dataset

The TaoWu dataset contains:

- 20 PD subjects
- 20 healthy controls
- 40 subjects in total

TaoWu is used for downstream classification and cross-validation.

### Functional Connectivity Representation

Each subject is represented by a **100 × 100 functional-connectivity matrix**, based on the Schaefer-100 atlas.

Each matrix is converted into a graph with:

- 100 nodes representing brain regions
- Identity-matrix node features
- Edges corresponding to region pairs
- Functional-connectivity coefficients as edge weights

The current graph-construction implementation uses the FC coefficients as signed edge weights; it does not automatically convert them to absolute values.

---

## 3. Overall Pipeline

```text
                  Functional-connectivity matrices
                               |
                               v
                     01_Prepare_Data.py
                               |
                               v
                       Graph-ready data
                               |
              +----------------+----------------+
              |                                 |
              v                                 v
      PPMI + Neurocon                         TaoWu
      External data                         Target data
              |                                 |
              v                                 |
       02_Train_GAE.py                          |
              |                                 |
              v                                 |
      Pretrained GAE encoder --------------------+
              |
              v
       03_Build_index.py
              |
              v
       FAISS knowledge base
              |
              v
     Retrieve K nearest neighbors
              |
              v
      Retrieval-based classifiers
              |
              v
         PD / HC prediction
```

The extended ablation compares classification with and without pretraining, retrieval and attention.

---

## 4. Graph Representation Learning

### Graph Autoencoder

The project uses a graph autoencoder to learn subject-level representations from functional-connectivity graphs.

The graph encoder contains:

```text
GCNConv
   |
ReLU
   |
Dropout
   |
GCNConv
   |
ReLU
   |
Dropout
   |
Linear projection
   |
Global mean pooling
   |
128-dimensional graph embedding
```

The encoder uses 64-dimensional hidden representations and produces a 128-dimensional graph-level embedding.

The graph autoencoder is trained on the external PPMI and Neurocon data using a reconstruction objective.

### Pretraining Configuration

The original pretraining script uses:

| Parameter | Value |
|---|---|
| Number of nodes | 100 |
| Hidden dimension | 64 |
| Embedding dimension | 128 |
| Epochs | 100 |
| Batch size | 32 |
| Learning rate | 0.0001 |
| Reconstruction loss | Mean squared error |

The pretrained encoder is saved as:

```text
gae_encoder.pth
```

---

## 5. External Knowledge Base and Retrieval

After pretraining, each external subject is encoded into a 128-dimensional embedding.

These embeddings are stored in a FAISS index using:

```python
faiss.IndexFlatL2(128)
```

The index performs exact nearest-neighbor search using Euclidean distance.

The main generated files are:

```text
knowledge_base.index
kb_embeddings.npy
kb_metadata_indexed.csv
```

For each TaoWu query subject, the retrieval module identifies the nearest subjects in the external knowledge base.

The extended experiments evaluate three retrieval-set sizes:

```text
K = 5
K = 10
K = 15
```

The primary six-model accuracy figure uses **K = 10** for the retrieval-based configurations.

---

## 6. Model Configurations

The extended ablation study compares six model families.

| Model | Configuration | Pretraining | Retrieval | Attention | Encoder training |
|---|---|---|---|---|---|
| **M0** | GNN from scratch | No | No | No | Trained on TaoWu |
| **M1** | Pretrained-only GNN | Yes | No | No | Frozen |
| **M2** | Mean retrieval | Yes | Yes | Equal-weight mean | Frozen |
| **M3** | RAC v1 | Yes | Yes | Dot-product | Frozen |
| **M4** | RAC v2 | Yes | Yes | Learnable query/key attention | Frozen |
| **M5** | RAC v2 with fine-tuning | Yes | Yes | Learnable query/key attention | Partially fine-tuned |

### M0 — GNN Trained from Scratch

The baseline model is initialized without external pretraining.

It learns directly from the TaoWu training subjects and does not use retrieved embeddings.

This establishes the performance of the target-data-only approach.

### M1 — Pretrained-Only Model

The pretrained graph encoder generates subject-level embeddings.

A downstream classifier is trained on the TaoWu training subjects.

No external neighbors are retrieved.

This configuration isolates the contribution of representation pretraining.

### M2 — Mean Retrieval

The target subject is encoded using the frozen pretrained encoder.

The K nearest external embeddings are retrieved from FAISS and combined using equal-weight averaging.

This model evaluates retrieval without a learned or similarity-weighted attention mechanism.

### M3 — RAC v1: Dot-Product Attention

The target embedding is used as a query to weight the retrieved embeddings using dot-product similarity.

The resulting retrieval context is combined with the target embedding for classification.

The pretrained encoder remains frozen.

### M4 — RAC v2: Learnable Attention

This configuration uses a learnable attention mechanism with query and key projections.

The pretrained encoder remains frozen.

It evaluates whether learnable attention improves retrieval-based classification.

### M5 — RAC v2 With Partial Fine-Tuning

This model uses learnable attention and partially fine-tunes the pretrained encoder.

In the current implementation:

- The first graph convolution layer is frozen.
- The second graph convolution layer is frozen.
- The final encoder projection layer (`lin_encode`) is trainable.

The external FAISS knowledge-base embeddings remain fixed during fine-tuning.

Consequently, the target query representation may change while the external index remains unchanged. This is an important implementation consideration when interpreting M5.

### Experiment Naming

The extended training code stores model configurations internally as E0–E5.

Their correspondence to the manuscript labels is:

```text
E0 -> M0
E1 -> M1
E2 -> M2
E3 -> M3
E4 -> M4
E5 -> M5
```

The E-labels are retained in the generated CSV files for reproducibility.

---

## 7. Model Training and Evaluation

### Original Experiments

The original repository includes both holdout and five-fold cross-validation implementations.

The original RAC experiments use K = 10 nearest neighbors.

The original five-fold scripts are:

```text
04_final1.py
04_final2.py
05_final.py
```

These scripts are retained to preserve the development history of the original framework.

The original single-model holdout evaluation is implemented separately in:

```text
06_Evaluate_models.py
```

The holdout evaluation script expects single-model checkpoints, while the five-fold training scripts save fold-specific checkpoints. Therefore, they should not be treated as a single interchangeable evaluation pipeline.

### Extended Ablation Study

The extended experiment is implemented in:

```text
experiment_06_improved_ablation_hpc.py
```

It evaluates:

- M0: GNN from scratch
- M1: Pretrained-only model
- M2–M5: Each evaluated at K = 5, 10 and 15

This produces **14 experimental conditions**.

### Cross-Validation Configuration

| Parameter | Value |
|---|---|
| Evaluation | Repeated stratified five-fold CV |
| Repeats | 5 |
| Validation folds per condition | 25 |
| Random seed | 42 |
| Maximum epochs | 100 |
| Batch size | 16 |
| Learning rate | 0.0001 |
| Weight decay | 0.0001 |
| Early-stopping patience | 15 |
| Retrieval-set sizes | 5, 10, 15 |

Each model is trained separately for each training/validation split.

### Evaluation Metrics

The experiment records:

- Classification accuracy
- ROC-AUC
- Sensitivity
- Specificity
- Precision
- F1-score

### Important Evaluation Limitation

In the current extended implementation, the outer cross-validation fold is also used to determine early stopping.

This means that the reported validation performance is influenced by model selection on the same fold and should not be interpreted as an unbiased nested-cross-validation estimate.

A more rigorous future evaluation would use an inner validation partition for model selection, with the outer fold reserved for final assessment.

The 25 repeated-CV folds also reuse subjects across repetitions. Fold results are therefore not 25 statistically independent samples.

---

## 8. Extended Ablation Results

The following table summarizes the current mean validation accuracies across 25 fold evaluations per condition.

These results correspond to the existing extended ablation outputs and should be regenerated if the evaluation protocol changes.

| Model | K | Mean accuracy (%) | SD (%) |
|---|---:|---:|---:|
| M0 — From scratch | — | 60.5 | 14.74 |
| M1 — Pretrained only | — | **66.0** | 14.67 |
| M2 — Mean retrieval | 5 | 61.5 | 12.46 |
| M2 — Mean retrieval | 10 | 59.5 | 18.50 |
| M2 — Mean retrieval | 15 | 62.5 | 16.54 |
| M3 — RAC v1 | 5 | 60.0 | 16.14 |
| M3 — RAC v1 | 10 | 58.5 | 16.82 |
| M3 — RAC v1 | 15 | 62.5 | 15.31 |
| M4 — RAC v2 frozen | 5 | 60.5 | 14.74 |
| M4 — RAC v2 frozen | 10 | 61.0 | 15.86 |
| M4 — RAC v2 frozen | 15 | 58.5 | 17.20 |
| M5 — RAC v2 fine-tuned | 5 | 60.5 | 15.60 |
| M5 — RAC v2 fine-tuned | 10 | 60.0 | 16.14 |
| M5 — RAC v2 fine-tuned | 15 | 56.5 | 16.97 |

### Main Observations

The pretrained-only model M1 achieved the highest mean accuracy in the current comparison.

Retrieval-based models did not consistently outperform M1 across the tested retrieval-set sizes.

Increasing K from 5 to 15 did not produce a consistent improvement across the retrieval-based configurations.

These observations suggest that pretrained graph representations may contribute more to classification than retrieval augmentation under the current experimental conditions.

The results do not establish that retrieval is generally ineffective; they are specific to the present datasets, retrieval method, architectures and evaluation protocol.

---

## 9. Neighbor-Composition Analysis

A separate analysis examines the diagnostic labels of subjects retrieved from the external knowledge base.

The analysis uses **K = 10** and the frozen pretrained encoder.

For each TaoWu subject, the procedure records:

1. The target subject's diagnosis (PD or HC).
2. The ten nearest external neighbors.
3. The number of retrieved PD subjects.
4. The number of retrieved HC subjects.
5. The proportions of PD and HC neighbors.

The analysis is intended to characterize the retrieval space, not to introduce another classification model.

### Group-Level Results

| Target group | Subjects | Mean PD neighbors (%) | Mean HC neighbors (%) |
|---|---:|---:|---:|
| PD | 20 | 79.5 ± 13.6 | 20.5 ± 13.6 |
| HC | 20 | 81.0 ± 12.9 | 19.0 ± 12.9 |

The retrieved diagnostic compositions were similar for PD and HC target subjects.

These proportions were also close to the external knowledge-base prevalence of approximately 82.2% PD.

The analysis therefore suggests limited diagnostic separation in the frozen encoder's retrieval neighborhoods.

Because the external knowledge base is class-imbalanced, this observation does not by itself establish the cause of the classification results.

### Illustrative Retrieved Neighborhoods

Examples observed in the neighbor-composition analysis include:

| Target ID | Diagnosis | PD neighbors | HC neighbors |
|---|---|---:|---:|
| `control032062` | HC | 6 | 4 |
| `patient032080` | PD | 6 | 4 |
| `patient032087` | PD | 10 | 0 |

These are subject-level examples rather than a separate test cohort.

**Data-sharing note:** Subject identifiers and per-subject retrieval files should only be published if permitted by the relevant dataset access terms. The group-level summary may be sufficient for a public manuscript.

The neighbor-composition script is maintained as:

```text
experiment_08_neighbor_composition.py
```

Its expected outputs include:

```text
neighbor_composition_per_subject.csv
neighbor_composition_manuscript_table.csv
```

The analysis must be run locally before these files are available.

---

## 10. Repository Structure

The repository contains the original training pipeline, additional experiments and publication-oriented analysis scripts.

```text
practical-project/
|
|-- 01_Prepare_Data.py
|-- 02_Train_GAE.py
|-- 03_Build_index.py
|
|-- 04_Train_rac_model.py
|-- 04_rac_cv.py
|-- 04_final1.py
|-- 04_final2.py
|
|-- 05_Train_baseline_model.py
|-- 05_final.py
|-- 06_Evaluate_models.py
|
|-- models.py
|-- utils.py
|
|-- experiment_01_pretrained_only.py
|-- experiment_02_retrieval_mean.py
|-- experiment_03_ablation_summary.py
|-- experiment_04_rac_v2_repeated_cv.py
|-- experiment_05_full_ablation_repeated_cv.py
|-- experiment_06_improved_ablation_hpc.py
|-- experiment07_figures.py
|
|-- README.md
|-- requirements.txt
|-- LICENSE
|-- .gitignore
```

### Extended Experiments

The repository includes additional experiments evaluating the effects of encoder pretraining, retrieval aggregation, attention mechanisms, retrieval-set size, and partial encoder fine-tuning.

The final ablation pipeline implements six model configurations (M0–M5), with repeated stratified cross-validation.

A separate retrieval-composition analysis examines the diagnostic distribution of nearest neighbors in the external knowledge base.

Earlier experimental implementations are retained for reproducibility and documentation of the development process.

---

## 11. Description of Main Python Files

### Data Preparation

**`01_Prepare_Data.py`**

Reads PPMI, Neurocon and TaoWu metadata and prepares graph-compatible functional-connectivity matrices.

The script excludes specified non-target diagnostic categories, converts matrices to NumPy arrays, and generates:

```text
master_metadata.csv
```

### Model Architectures

**`models.py`**

Defines:

- `GAEEncoder`
- `GraphAutoencoder`
- `AttentionMechanism`
- `AttentionMechanismLinear`
- `RAC_Model`
- `Baseline_GNN`

### Graph Utilities

**`utils.py`**

Provides the `FCDataset` class for converting FC matrices into graph objects and includes FAISS-loading utilities.

### Encoder Pretraining

**`02_Train_GAE.py`**

Trains the graph autoencoder on the external knowledge-base subjects.

### Knowledge-Base Construction

**`03_Build_index.py`**

Generates the external subject embeddings and creates the FAISS nearest-neighbor index.

### Original RAC Training

**`04_Train_rac_model.py`**

Implements the original RAC training procedure using a stratified holdout split.

**`04_final1.py`**

Implements five-fold cross-validation with dot-product attention.

**`04_final2.py`**

Implements five-fold cross-validation with learnable linear attention.

**`04_rac_cv.py`**

Contains an additional RAC cross-validation implementation from the original project.

### Original Baseline Training

**`05_Train_baseline_model.py`**

Trains the GNN baseline without external retrieval.

**`05_final.py`**

Implements five-fold cross-validation for the baseline.

### Original Model Evaluation

**`06_Evaluate_models.py`**

Evaluates the original holdout models and reports classification metrics. Its checkpoint expectations differ from those of the fold-specific training scripts.

### Extended Experiments

**`experiment_01_pretrained_only.py`**

Earlier pretrained-only experiment.

**`experiment_02_retrieval_mean.py`**

Earlier mean-retrieval experiment.

**`experiment_03_ablation_summary.py`**

Earlier ablation-result summary.

**`experiment_04_rac_v2_repeated_cv.py`**

Repeated-CV experiment focused on RAC v2.

**`experiment_05_full_ablation_repeated_cv.py`**

Earlier multi-model ablation implementation.

**`experiment_06_improved_ablation_hpc.py`**

Current extended experiment comparing M0–M5, including K=5, 10 and 15 for retrieval-based models.

**`experiment07_figures.py`**

Generates scientific figures and comparisons from the extended ablation results.

**`experiment_08_neighbor_composition.py`**

Locally maintained script for analyzing PD/HC diagnostic composition among retrieved external neighbors.

---

## 12. Installation

Use a Python environment with compatible versions of PyTorch, PyTorch Geometric and FAISS.

The core dependencies include:

```text
numpy
pandas
scipy
scikit-learn
torch
torch-geometric
faiss-cpu
matplotlib
```

Install the dependencies with:

```bash
pip install -r requirements.txt
```

The repository's `requirements.txt` must include `matplotlib` for the figure-generation script.

For reproducibility, a future release should pin tested dependency versions.

---

## 13. Running the Pipeline

### Step 1 — Prepare the Data

```bash
python 01_Prepare_Data.py
```

### Step 2 — Train the Graph Autoencoder

```bash
python 02_Train_GAE.py
```

### Step 3 — Build the External Knowledge Base

```bash
python 03_Build_index.py
```

### Step 4 — Run Original Five-Fold Experiments

For RAC v1 and RAC v2:

```bash
python 04_final1.py
python 04_final2.py
```

For the baseline model:

```bash
python 05_final.py
```

These experiments use the original training protocol and should be distinguished from the extended repeated-CV experiments.

### Step 5 — Run the Extended Ablation Study

```bash
python experiment_06_improved_ablation_hpc.py
```

The script expects the processed metadata, pretrained encoder and knowledge-base files to be present.

Its current path configuration is designed for a particular HPC directory structure and must be adapted if running in another environment.

Main output files:

```text
experiment_results/
    sana06_all_folds.csv
    sana06_summary.csv
    sana06_predictions.csv
```

### Step 6 — Generate Publication Figures

```bash
python experiment07_figures.py
```

The script reads `sana06_all_folds.csv` and produces scientific plots in PDF, PNG and SVG formats.

The locally updated version includes a six-model comparison figure:

```text
Figure7_ISBI_M0_to_M5_accuracy.pdf
```

This plot compares M0–M5, with K=10 for the retrieval-based models.

Other figures examine:

- Accuracy distributions
- ROC-AUC
- Sensitivity and specificity
- Effect of retrieval-set size
- Combined ablation results

**Note:** The figure script currently uses local output-directory paths that may need to be updated for another computer or HPC environment.

### Step 7 — Analyze Retrieved Neighbors

After adding the local script to the repository:

```bash
python experiment_08_neighbor_composition.py
```

The script generates subject-level retrieval composition and a manuscript-oriented summary table.

Its required input paths must be configured for the local processed-data directory.

---

## 14. Reproducibility and Data Management

The project uses locally stored processed neuroimaging data and trained model artifacts.

These should be managed separately from the public source-code repository.

Typical local artifacts include:

```text
processed_data_1/
processed_data_2/
gae_encoder.pth
knowledge_base.index
kb_embeddings.npy
master_metadata.csv
experiment_results/
```

Data and derived subject-level files should only be shared in accordance with the applicable dataset access conditions.

The following directories should normally be excluded from Git tracking:

```gitignore
processed_data_1/
processed_data_2/
venv/
.venv/
__pycache__/
*.py[cod]
```

Some original scripts contain machine-specific Windows paths, while the extended experiment contains HPC-specific paths.

These paths must be adjusted to reproduce the workflow in another computing environment.

---

## 15. Interpretation and Limitations

The current experiments suggest that pretrained graph representations can improve classification relative to training a GNN from scratch.

However, retrieval augmentation and attention mechanisms did not consistently improve mean accuracy over the pretrained-only configuration.

Several limitations should be considered:

1. **Small target cohort:** Only 40 TaoWu subjects are used for downstream classification.
2. **Knowledge-base class imbalance:** The external knowledge base contains substantially more PD than HC subjects.
3. **Cross-validation model selection:** The current repeated-CV implementation uses the reported outer validation folds for early stopping.
4. **Overlapping repeated-CV folds:** Validation observations across repeats are not independent.
5. **Fixed retrieval embeddings during M5 training:** Partial encoder fine-tuning may change query representations without updating external FAISS embeddings.
6. **Retrieval-composition analysis:** Similar PD/HC neighborhood proportions are descriptive and do not establish a causal explanation for classification performance.

The results should therefore be interpreted as an ablation study of the current implementation rather than definitive evidence against retrieval augmentation.

---

## 16. Research Summary

This project studies Parkinson's disease classification using graph representations of resting-state functional connectivity.

The main contributions of the extended investigation are:

- A comparison of training from scratch versus external representation pretraining
- An evaluation of mean retrieval, dot-product attention and learnable attention
- A comparison of frozen versus partially fine-tuned pretrained encoders
- An investigation of retrieval-set sizes K=5, 10 and 15
- A diagnostic analysis of the retrieved external neighborhoods

Under the current experimental setup, the pretrained-only model achieved the highest mean validation accuracy, while retrieval-based configurations did not provide a consistent additional benefit.

The neighbor-composition analysis further suggests that the frozen embedding space does not strongly separate PD and HC subjects through nearest-neighbor diagnostic composition.

Future work should prioritize improved model-selection procedures, class-balanced retrieval, better alignment of fine-tuned query and knowledge-base embeddings, and validation using larger independent cohorts.

---

## 17. Project Status

The repository contains the original practical-project implementation and additional experimental work for an ISBI-oriented scientific manuscript.

The original pipeline and the extended ablation have different evaluation protocols; their reported accuracies should not be combined into a single direct comparison.

The extended results, figures and neighbor-composition analysis are being consolidated for reproducible documentation and manuscript preparation.

---

## License

See the repository's `LICENSE` file for licensing details.
