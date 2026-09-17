# ============================================================
# sana_01_pretrained_only.py
#
# ABLATION 1:
# External GAE pretraining only
#
# Uses:
#   - GAE encoder pretrained on PPMI + Neurocon
#   - TaoWu for supervised classification
#
# Does NOT use:
#   - FAISS retrieval
#   - retrieved neighbours
#   - attention
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
#
# IMPORTANT:
# Change only output_dir if your processed_data_1 directory
# is somewhere else.
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

# Separate folder for Sana's analyses
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
# 4. Load TaoWu data
# ------------------------------------------------------------

df = pd.read_csv(master_csv)

df_target = (
    df[df["dataset_source"] == "TaoWu"]
    .reset_index(drop=True)
)

print("\nTotal TaoWu subjects:", len(df_target))
print(df_target["label"].value_counts())


# ------------------------------------------------------------
# 5. SAME 80/20 split as original repository
#
# random_state=42 and stratification match the original scripts.
# ------------------------------------------------------------

df_train, df_test = train_test_split(
    df_target,
    test_size=0.20,
    stratify=df_target["label"],
    random_state=42,
)

df_train = df_train.reset_index(drop=True)
df_test = df_test.reset_index(drop=True)

print("\nTraining subjects:", len(df_train))
print("Test subjects:", len(df_test))


# Save exact split used by this experiment
df_train.to_csv(
    os.path.join(
        results_dir,
        "sana_pretrained_only_train_subjects.csv"
    ),
    index=False,
)

df_test.to_csv(
    os.path.join(
        results_dir,
        "sana_pretrained_only_test_subjects.csv"
    ),
    index=False,
)


# ------------------------------------------------------------
# 6. DataLoaders
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
# 7. Define classifier
#
# Important:
# The pretrained encoder is frozen.
#
# Only the classification head learns from TaoWu.
# ------------------------------------------------------------

class PretrainedOnlyClassifier(nn.Module):

    def __init__(
        self,
        gae_encoder,
        embedding_dim=128
    ):
        super().__init__()

        self.gae_encoder = gae_encoder

        self.classification_head = nn.Sequential(
            nn.Linear(embedding_dim, 64),
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
    ):

        # Encoder is frozen.
        # no_grad prevents unnecessary gradient calculations.
        with torch.no_grad():
            embedding = self.gae_encoder(
                x,
                edge_index,
                edge_weight,
                batch,
            )

        prediction = self.classification_head(
            embedding
        )

        return prediction


# ------------------------------------------------------------
# 8. Load pretrained GAE encoder
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

# Freeze encoder parameters
for parameter in pretrained_gae.encoder.parameters():
    parameter.requires_grad = False

pretrained_gae.encoder.eval()


# ------------------------------------------------------------
# 9. Initialize model
# ------------------------------------------------------------

model = PretrainedOnlyClassifier(
    gae_encoder=pretrained_gae.encoder,
    embedding_dim=128,
).to(device)


# Only optimize trainable parameters
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

NUM_EPOCHS = 100


# ------------------------------------------------------------
# 10. Training loop
# ------------------------------------------------------------

print(
    "\nStarting PRETRAINED-ONLY classifier training..."
)

training_history = []

best_train_loss = float("inf")


for epoch in range(NUM_EPOCHS):

    model.train()

    # Keep frozen GAE encoder in evaluation mode
    # even though the classifier itself is training.
    model.gae_encoder.eval()

    total_train_loss = 0.0

    for data in train_loader:

        data = data.to(device)

        optimizer.zero_grad()

        predictions = model(
            data.x,
            data.edge_index,
            data.edge_weight,
            data.batch,
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
                "sana_pretrained_only_model.pth"
            )
        )

    if (epoch + 1) % 10 == 0:

        print(
            f"Epoch {epoch + 1:03d}/{NUM_EPOCHS} "
            f"| Train Loss: {avg_train_loss:.4f}"
        )


# ------------------------------------------------------------
# 11. Save training history
# ------------------------------------------------------------

pd.DataFrame(training_history).to_csv(
    os.path.join(
        results_dir,
        "sana_pretrained_only_training_history.csv"
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
            "sana_pretrained_only_model.pth"
        ),
        map_location=device,
    )
)

model.eval()
model.gae_encoder.eval()


# ------------------------------------------------------------
# 13. Evaluate on TaoWu held-out test subjects
# ------------------------------------------------------------

all_labels = []
all_probs = []
all_predictions = []
all_subject_ids = []


with torch.no_grad():

    for batch_index, data in enumerate(test_loader):

        data = data.to(device)

        probability = model(
            data.x,
            data.edge_index,
            data.edge_weight,
            data.batch,
        )

        probability = probability.view(-1).item()

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
# 15. Save subject-level predictions
# ------------------------------------------------------------

prediction_df = pd.DataFrame({

    "subject_id": all_subject_ids,

    "ground_truth": all_labels,

    "probability_PD": all_probs,

    "predicted_class": all_predictions,
})

prediction_df.to_csv(
    os.path.join(
        results_dir,
        "sana_pretrained_only_predictions.csv"
    ),
    index=False,
)


# ------------------------------------------------------------
# 16. Save summary result
# ------------------------------------------------------------

summary_df = pd.DataFrame([{

    "experiment": "Pretrained encoder only",

    "external_pretraining": "Yes",

    "retrieval": "No",

    "attention": "No",

    "n_train": len(df_train),

    "n_test": len(df_test),

    "accuracy": accuracy,

    "roc_auc": auc,

    "precision": precision,

    "recall_sensitivity": recall,

    "specificity": specificity,

    "f1_score": f1,

    "TN": tn,
    "FP": fp,
    "FN": fn,
    "TP": tp,
}])

summary_df.to_csv(
    os.path.join(
        results_dir,
        "sana_pretrained_only_summary.csv"
    ),
    index=False,
)


# ------------------------------------------------------------
# 17. Print results
# ------------------------------------------------------------

print("\n========================================")
print("PRETRAINED-ONLY RESULTS")
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