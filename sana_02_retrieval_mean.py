# ============================================================
# sana_02_retrieval_mean.py
#
# ABLATION 2:
# Pretrained encoder + FAISS retrieval
# WITHOUT attention
#
# The retrieved top-k embeddings are simply averaged.
#
# Saves all outputs separately in:
#   sana_results/
# ============================================================

import os
import random
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import faiss

from torch_geometric.loader import DataLoader
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    roc_auc_score,
    confusion_matrix,
    precision_score,
    recall_score,
    f1_score,
)

from models import GraphAutoencoder
from utils import FCDataset


# ------------------------------------------------------------
# 1. Reproducibility
# ------------------------------------------------------------

SEED = 42

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)


# ------------------------------------------------------------
# 2. Paths
# ------------------------------------------------------------

output_dir = r'C:\Users\imam\Desktop\Tara Project\processed_data_1'

master_csv = os.path.join(
    output_dir,
    "master_metadata.csv"
)

encoder_weights = os.path.join(
    output_dir,
    "gae_encoder.pth"
)

index_path = os.path.join(
    output_dir,
    "knowledge_base.index"
)

kb_embeddings_path = os.path.join(
    output_dir,
    "kb_embeddings.npy"
)

kb_metadata_path = os.path.join(
    output_dir,
    "kb_metadata_indexed.csv"
)

results_dir = os.path.join(
    output_dir,
    "sana_results"
)

os.makedirs(results_dir, exist_ok=True)


# ------------------------------------------------------------
# 3. Device
# ------------------------------------------------------------

device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

print("Using device:", device)


# ------------------------------------------------------------
# 4. Load TaoWu
# ------------------------------------------------------------

df = pd.read_csv(master_csv)

# Fix npy paths from Tara's computer to my local computer
df["npy_path"] = df["npy_path"].apply(
    lambda x: os.path.join(
        output_dir,
        os.path.basename(x)
    )
)

df_target = (
    df[df["dataset_source"] == "TaoWu"]
    .reset_index(drop=True)
)


# SAME split as original scripts
df_train, df_test = train_test_split(
    df_target,
    test_size=0.20,
    stratify=df_target["label"],
    random_state=42,
)

df_train = df_train.reset_index(drop=True)
df_test = df_test.reset_index(drop=True)

print("\nTotal TaoWu subjects:", len(df_target))
print("Training subjects:", len(df_train))
print("Test subjects:", len(df_test))


# Save the exact split
df_train.to_csv(
    os.path.join(
        results_dir,
        "sana_retrieval_mean_train_subjects.csv"
    ),
    index=False,
)

df_test.to_csv(
    os.path.join(
        results_dir,
        "sana_retrieval_mean_test_subjects.csv"
    ),
    index=False,
)


# ------------------------------------------------------------
# 5. DataLoaders
# ------------------------------------------------------------

train_loader = DataLoader(
    FCDataset(df_train),
    batch_size=16,
    shuffle=True,
)

test_loader = DataLoader(
    FCDataset(df_test),
    batch_size=1,
    shuffle=False,
)


# ------------------------------------------------------------
# 6. Load pretrained GAE
# ------------------------------------------------------------

pretrained_gae = GraphAutoencoder(
    num_nodes=100,
    input_dim=100,
    hidden_dim=64,
    embedding_dim=128,
)

pretrained_gae.encoder.load_state_dict(
    torch.load(
        encoder_weights,
        map_location=device
    )
)

pretrained_gae.encoder.to(device)

for parameter in pretrained_gae.encoder.parameters():
    parameter.requires_grad = False

pretrained_gae.encoder.eval()


# ------------------------------------------------------------
# 7. Load FAISS knowledge base
# ------------------------------------------------------------

index = faiss.read_index(
    index_path
)

kb_embeddings = np.load(
    kb_embeddings_path
).astype("float32")

kb_metadata = pd.read_csv(
    kb_metadata_path
)

K = 10

print("\nFAISS knowledge-base size:", index.ntotal)
print("K =", K)


# ------------------------------------------------------------
# 8. Define retrieval model WITHOUT attention
# ------------------------------------------------------------

class RetrievalMeanClassifier(nn.Module):

    def __init__(
        self,
        gae_encoder,
        embedding_dim=128
    ):
        super().__init__()

        self.gae_encoder = gae_encoder

        self.classification_head = nn.Sequential(
            nn.Linear(
                embedding_dim * 2,
                64
            ),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, 1),
            nn.Sigmoid(),
        )

    def forward(
        self,
        x,
        edge_index,
        edge_weight,
        batch,
        retrieved_embeddings,
    ):

        # Frozen query encoder
        with torch.no_grad():

            query_embedding = self.gae_encoder(
                x,
                edge_index,
                edge_weight,
                batch,
            )

        # --------------------------------------
        # NO ATTENTION
        #
        # All retrieved neighbours receive
        # exactly equal importance.
        # --------------------------------------

        context_embedding = (
            retrieved_embeddings.mean(dim=1)
        )

        # Query:   [batch, 128]
        # Context: [batch, 128]
        #
        # Combined:
        # [batch, 256]

        augmented_embedding = torch.cat(
            (
                query_embedding,
                context_embedding,
            ),
            dim=1,
        )

        prediction = self.classification_head(
            augmented_embedding
        )

        return prediction


# ------------------------------------------------------------
# 9. Initialize model
# ------------------------------------------------------------

model = RetrievalMeanClassifier(
    gae_encoder=pretrained_gae.encoder,
    embedding_dim=128,
).to(device)


optimizer = torch.optim.Adam(
    filter(
        lambda p: p.requires_grad,
        model.parameters()
    ),
    lr=0.001,
)

scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer,
    mode="min",
    factor=0.5,
    patience=3,
)

loss_fn = torch.nn.BCELoss()

NUM_EPOCHS = 200


# ------------------------------------------------------------
# Helper:
# retrieve top-k KB embeddings for a batch
# ------------------------------------------------------------

def retrieve_embeddings(
    query_embedding,
):

    query_np = (
        query_embedding
        .detach()
        .cpu()
        .numpy()
        .astype("float32")
    )

    distances, neighbour_indices = (
        index.search(
            query_np,
            K
        )
    )

    retrieved_np = (
        kb_embeddings[
            neighbour_indices
        ]
    )

    retrieved_tensor = (
        torch
        .from_numpy(retrieved_np)
        .float()
        .to(device)
    )

    return (
        retrieved_tensor,
        distances,
        neighbour_indices,
    )


# ------------------------------------------------------------
# 10. Training
# ------------------------------------------------------------

print(
    "\nStarting RETRIEVAL-MEAN training..."
)

training_history = []

best_train_loss = float("inf")


for epoch in range(NUM_EPOCHS):

    model.train()

    # Frozen encoder must remain eval
    model.gae_encoder.eval()

    total_train_loss = 0.0

    for data in train_loader:

        data = data.to(device)

        optimizer.zero_grad()

        # First create the query embedding for FAISS
        with torch.no_grad():

            query_embedding = (
                model.gae_encoder(
                    data.x,
                    data.edge_index,
                    data.edge_weight,
                    data.batch,
                )
            )

        # Retrieve top-K external subjects
        retrieved_embeddings, _, _ = (
            retrieve_embeddings(
                query_embedding
            )
        )

        predictions = model(
            data.x,
            data.edge_index,
            data.edge_weight,
            data.batch,
            retrieved_embeddings,
        )

        loss = loss_fn(
            predictions.view(-1),
            data.y.float().view(-1),
        )

        loss.backward()

        torch.nn.utils.clip_grad_norm_(
            model.classification_head.parameters(),
            max_norm=1.0,
        )

        optimizer.step()

        total_train_loss += loss.item()

    avg_train_loss = (
        total_train_loss / len(train_loader)
    )

    scheduler.step(avg_train_loss)

    training_history.append({
        "epoch": epoch + 1,
        "train_loss": avg_train_loss,
    })

    if avg_train_loss < best_train_loss:

        best_train_loss = avg_train_loss

        torch.save(
            model.state_dict(),
            os.path.join(
                results_dir,
                "sana_retrieval_mean_model.pth"
            )
        )

    if (epoch + 1) % 10 == 0:

        print(
            f"Epoch {epoch + 1:03d}/{NUM_EPOCHS}"
            f" | Train Loss: "
            f"{avg_train_loss:.4f}"
        )


# ------------------------------------------------------------
# 11. Save training history
# ------------------------------------------------------------

pd.DataFrame(training_history).to_csv(
    os.path.join(
        results_dir,
        "sana_retrieval_mean_training_history.csv"
    ),
    index=False,
)


# ------------------------------------------------------------
# 12. Load best model
# ------------------------------------------------------------

model.load_state_dict(
    torch.load(
        os.path.join(
            results_dir,
            "sana_retrieval_mean_model.pth"
        ),
        map_location=device,
    )
)

model.eval()
model.gae_encoder.eval()


# ------------------------------------------------------------
# 13. Test evaluation
# ------------------------------------------------------------

all_subject_ids = []
all_labels = []
all_probs = []
all_predictions = []

retrieval_rows = []


with torch.no_grad():

    for batch_index, data in enumerate(test_loader):

        data = data.to(device)

        query_embedding = (
            model.gae_encoder(
                data.x,
                data.edge_index,
                data.edge_weight,
                data.batch,
            )
        )

        (
            retrieved_embeddings,
            distances,
            neighbour_indices,
        ) = retrieve_embeddings(
            query_embedding
        )

        probability = model(
            data.x,
            data.edge_index,
            data.edge_weight,
            data.batch,
            retrieved_embeddings,
        )

        probability = (
            probability
            .view(-1)
            .item()
        )

        prediction = (
            1 if probability > 0.5 else 0
        )

        label = int(
            data.y.view(-1).item()
        )

        subject_id = (
            df_test.iloc[batch_index]["subject_id"]
        )

        all_subject_ids.append(subject_id)
        all_labels.append(label)
        all_probs.append(probability)
        all_predictions.append(prediction)

        # --------------------------------------
        # Save retrieval details for every
        # TaoWu test subject
        # --------------------------------------

        for rank in range(K):

            kb_index = int(
                neighbour_indices[0][rank]
            )

            kb_row = (
                kb_metadata.iloc[kb_index]
            )

            retrieval_rows.append({

                "query_subject_id":
                    subject_id,

                "query_ground_truth":
                    label,

                "rank":
                    rank + 1,

                "faiss_distance":
                    float(
                        distances[0][rank]
                    ),

                "retrieved_kb_index":
                    kb_index,

                "retrieved_subject_id":
                    kb_row.get(
                        "subject_id",
                        ""
                    ),

                "retrieved_label":
                    kb_row.get(
                        "label",
                        ""
                    ),

                "retrieved_diagnosis":
                    kb_row.get(
                        "diagnosis",
                        ""
                    ),

                "retrieved_dataset":
                    kb_row.get(
                        "dataset_source",
                        ""
                    ),
            })


# ------------------------------------------------------------
# 14. Metrics
# ------------------------------------------------------------

accuracy = accuracy_score(
    all_labels,
    all_predictions,
)

precision = precision_score(
    all_labels,
    all_predictions,
    zero_division=0,
)

recall = recall_score(
    all_labels,
    all_predictions,
    zero_division=0,
)

f1 = f1_score(
    all_labels,
    all_predictions,
    zero_division=0,
)

try:

    auc = roc_auc_score(
        all_labels,
        all_probs,
    )

except ValueError:

    auc = np.nan


cm = confusion_matrix(
    all_labels,
    all_predictions,
)

tn, fp, fn, tp = cm.ravel()

specificity = (
    tn / (tn + fp)
    if (tn + fp) > 0
    else np.nan
)


# ------------------------------------------------------------
# 15. Save predictions
# ------------------------------------------------------------

prediction_df = pd.DataFrame({

    "subject_id":
        all_subject_ids,

    "ground_truth":
        all_labels,

    "probability_PD":
        all_probs,

    "predicted_class":
        all_predictions,
})

prediction_df.to_csv(
    os.path.join(
        results_dir,
        "sana_retrieval_mean_predictions.csv"
    ),
    index=False,
)


# ------------------------------------------------------------
# 16. Save actual retrieval information
#
# This is useful later for analysing:
# - how many PD neighbours were retrieved
# - PPMI vs Neurocon retrieval
# - neighbour distances
# ------------------------------------------------------------

retrieval_df = pd.DataFrame(
    retrieval_rows
)

retrieval_df.to_csv(
    os.path.join(
        results_dir,
        "sana_retrieval_mean_neighbors.csv"
    ),
    index=False,
)


# ------------------------------------------------------------
# 17. Save summary
# ------------------------------------------------------------

summary_df = pd.DataFrame([{

    "experiment":
        "Pretrained + FAISS retrieval mean",

    "external_pretraining":
        "Yes",

    "retrieval":
        "Yes",

    "attention":
        "No",

    "retrieval_method":
        "Equal mean of top-10 neighbours",

    "K":
        K,

    "n_train":
        len(df_train),

    "n_test":
        len(df_test),

    "accuracy":
        accuracy,

    "roc_auc":
        auc,

    "precision":
        precision,

    "recall_sensitivity":
        recall,

    "specificity":
        specificity,

    "f1_score":
        f1,

    "TN":
        tn,

    "FP":
        fp,

    "FN":
        fn,

    "TP":
        tp,
}])

summary_df.to_csv(
    os.path.join(
        results_dir,
        "sana_retrieval_mean_summary.csv"
    ),
    index=False,
)


# ------------------------------------------------------------
# 18. Print results
# ------------------------------------------------------------

print("\n========================================")
print("RETRIEVAL-MEAN RESULTS")
print("========================================")

print(f"Accuracy:    {accuracy * 100:.2f}%")
print(f"ROC-AUC:     {auc:.4f}")
print(f"Precision:   {precision:.4f}")
print(f"Sensitivity: {recall:.4f}")
print(f"Specificity: {specificity:.4f}")
print(f"F1-score:    {f1:.4f}")

print("\nConfusion Matrix:")
print(cm)

print(
    "\nResults saved in:",
    results_dir
)