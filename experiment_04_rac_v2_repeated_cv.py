# ============================================================
# SANA 04: RAC v2 - REPEATED STRATIFIED 5-FOLD CROSS VALIDATION
# ============================================================
#
# Purpose:
# Evaluate RAC v2 (Linear Attention) more robustly on TaoWu.
#
# Method:
# - 5-fold stratified CV
# - repeated 10 times
# - total = 50 validation folds
# - frozen pretrained GAE encoder
# - FAISS retrieval, K = 10
# - linear attention
# - early stopping
#
# Original Tara files are NOT modified.
# ============================================================

import os
import random
import numpy as np
import pandas as pd
import torch
import faiss

from torch_geometric.loader import DataLoader

from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.metrics import (
    accuracy_score,
    roc_auc_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix
)

from models import (
    GraphAutoencoder,
    AttentionMechanismLinear,
    RAC_Model
)

from utils import FCDataset


# ============================================================
# 1. REPRODUCIBILITY
# ============================================================

SEED = 42

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)


# ============================================================
# 2. DEVICE
# ============================================================

device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

print(f"Using device: {device}")


# ============================================================
# 3. PATHS
# ============================================================

output_dir = r"C:\Users\imam\Desktop\Tara Project\processed_data_1"

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

results_dir = os.path.join(
    output_dir,
    "sana_results"
)

os.makedirs(
    results_dir,
    exist_ok=True
)


# ============================================================
# 4. LOAD METADATA
# ============================================================

df = pd.read_csv(master_csv)

# Fix Tara's old absolute npy paths
df["npy_path"] = df["npy_path"].apply(
    lambda x: os.path.join(
        output_dir,
        os.path.basename(x)
    )
)

df_target = df[
    df["dataset_source"] == "TaoWu"
].reset_index(drop=True)

print("\nTaoWu subjects:")
print(df_target["label"].value_counts().sort_index())

print(f"\nTotal TaoWu subjects: {len(df_target)}")


# ============================================================
# 5. LOAD PRETRAINED GAE ENCODER
# ============================================================

base_gae = GraphAutoencoder(
    num_nodes=100,
    input_dim=100,
    hidden_dim=64,
    embedding_dim=128
)

base_gae.encoder.load_state_dict(
    torch.load(
        encoder_weights,
        map_location=device
    )
)

base_gae.encoder.to(device)

# Freeze encoder
for param in base_gae.encoder.parameters():
    param.requires_grad = False

# Very important:
# keep encoder in evaluation mode
base_gae.encoder.eval()


# ============================================================
# 6. LOAD FAISS KNOWLEDGE BASE
# ============================================================

index = faiss.read_index(index_path)

kb_embeddings = np.load(
    kb_embeddings_path
).astype("float32")

K = 10


# ============================================================
# 7. CROSS-VALIDATION SETTINGS
# ============================================================

N_SPLITS = 5
N_REPEATS = 10

cv = RepeatedStratifiedKFold(
    n_splits=N_SPLITS,
    n_repeats=N_REPEATS,
    random_state=SEED
)

print(
    f"\nStarting Repeated Stratified "
    f"{N_SPLITS}-Fold CV"
)

print(
    f"Repeats: {N_REPEATS}"
)

print(
    f"Total validation folds: "
    f"{N_SPLITS * N_REPEATS}"
)


# ============================================================
# 8. TRAINING SETTINGS
# ============================================================

NUM_EPOCHS = 100

LEARNING_RATE = 0.0001

WEIGHT_DECAY = 1e-4

EARLY_STOPPING_PATIENCE = 15

BATCH_SIZE = 16


# ============================================================
# 9. RESULT STORAGE
# ============================================================

all_fold_results = []

all_predictions = []


# ============================================================
# 10. REPEATED CROSS-VALIDATION LOOP
# ============================================================

for fold_number, (
    train_idx,
    val_idx
) in enumerate(
    cv.split(
        df_target,
        df_target["label"]
    ),
    start=1
):

    repeat_number = (
        (fold_number - 1) // N_SPLITS
    ) + 1

    fold_in_repeat = (
        (fold_number - 1) % N_SPLITS
    ) + 1

    print("\n" + "=" * 60)

    print(
        f"Repeat {repeat_number}/{N_REPEATS} "
        f"| Fold {fold_in_repeat}/{N_SPLITS}"
    )

    print("=" * 60)

    # --------------------------------------------------------
    # Split data
    # --------------------------------------------------------

    df_train = (
        df_target
        .iloc[train_idx]
        .reset_index(drop=True)
    )

    df_val = (
        df_target
        .iloc[val_idx]
        .reset_index(drop=True)
    )

    train_dataset = FCDataset(df_train)

    val_dataset = FCDataset(df_val)

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=1,
        shuffle=False
    )


    # --------------------------------------------------------
    # Fresh RAC v2 model for every fold
    # --------------------------------------------------------

    attention_model = AttentionMechanismLinear(
        embedding_dim=128
    )

    rac_model = RAC_Model(
        gae_encoder=base_gae.encoder,
        attention_model=attention_model,
        embedding_dim=128
    )

    rac_model.to(device)


    # --------------------------------------------------------
    # Optimizer / loss
    # --------------------------------------------------------

    optimizer = torch.optim.Adam(
        filter(
            lambda p: p.requires_grad,
            rac_model.parameters()
        ),
        lr=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY
    )

    scheduler = (
        torch.optim.lr_scheduler
        .ReduceLROnPlateau(
            optimizer,
            mode="min",
            factor=0.5,
            patience=5
        )
    )

    loss_fn = torch.nn.BCELoss()


    # --------------------------------------------------------
    # Early stopping variables
    # --------------------------------------------------------

    best_val_loss = float("inf")

    best_epoch = 0

    best_model_state = None

    patience_counter = 0


    # ========================================================
    # 11. TRAINING LOOP
    # ========================================================

    for epoch in range(
        1,
        NUM_EPOCHS + 1
    ):

        rac_model.train()

        # Important:
        # RAC model is training,
        # but pretrained encoder must remain eval
        rac_model.gae_encoder.eval()

        total_train_loss = 0.0


        # ----------------------------------------------------
        # Training batches
        # ----------------------------------------------------

        for data in train_loader:

            data = data.to(device)

            optimizer.zero_grad()


            # Query embedding from frozen encoder
            with torch.no_grad():

                v_query = (
                    rac_model.gae_encoder(
                        data.x,
                        data.edge_index,
                        data.edge_weight,
                        data.batch
                    )
                )


            # FAISS retrieval
            query_np = (
                v_query
                .detach()
                .cpu()
                .numpy()
                .astype("float32")
            )

            _, neighbor_indices = (
                index.search(
                    query_np,
                    K
                )
            )

            v_retrieved = (
                torch.from_numpy(
                    kb_embeddings[
                        neighbor_indices
                    ]
                )
                .float()
                .to(device)
            )


            # RAC forward pass
            predictions, _ = rac_model(
                data.x,
                data.edge_index,
                data.edge_weight,
                data.batch,
                v_retrieved
            )


            labels = (
                data.y
                .float()
                .view(-1)
            )

            predictions = (
                predictions
                .view(-1)
            )


            loss = loss_fn(
                predictions,
                labels
            )

            loss.backward()

            torch.nn.utils.clip_grad_norm_(
                rac_model.parameters(),
                max_norm=1.0
            )

            optimizer.step()

            total_train_loss += (
                loss.item()
            )


        avg_train_loss = (
            total_train_loss /
            len(train_loader)
        )


        # ====================================================
        # 12. VALIDATION
        # ====================================================

        rac_model.eval()
        rac_model.gae_encoder.eval()

        total_val_loss = 0.0

        val_labels = []
        val_probs = []


        with torch.no_grad():

            for data in val_loader:

                data = data.to(device)


                v_query = (
                    rac_model.gae_encoder(
                        data.x,
                        data.edge_index,
                        data.edge_weight,
                        data.batch
                    )
                )


                query_np = (
                    v_query
                    .cpu()
                    .numpy()
                    .astype("float32")
                )


                _, neighbor_indices = (
                    index.search(
                        query_np,
                        K
                    )
                )


                v_retrieved = (
                    torch.from_numpy(
                        kb_embeddings[
                            neighbor_indices
                        ]
                    )
                    .float()
                    .to(device)
                )


                predictions, _ = rac_model(
                    data.x,
                    data.edge_index,
                    data.edge_weight,
                    data.batch,
                    v_retrieved
                )


                labels = (
                    data.y
                    .float()
                    .view(-1)
                )

                predictions = (
                    predictions
                    .view(-1)
                )


                loss = loss_fn(
                    predictions,
                    labels
                )

                total_val_loss += (
                    loss.item()
                )


                val_labels.extend(
                    labels
                    .cpu()
                    .numpy()
                    .tolist()
                )

                val_probs.extend(
                    predictions
                    .cpu()
                    .numpy()
                    .tolist()
                )


        avg_val_loss = (
            total_val_loss /
            len(val_loader)
        )


        scheduler.step(
            avg_val_loss
        )


        # ----------------------------------------------------
        # Early stopping
        # ----------------------------------------------------

        if avg_val_loss < best_val_loss:

            best_val_loss = (
                avg_val_loss
            )

            best_epoch = epoch

            patience_counter = 0

            # Save model in memory
            best_model_state = {
                key:
                    value.detach()
                    .cpu()
                    .clone()

                for key, value
                in rac_model
                .state_dict()
                .items()
            }

        else:

            patience_counter += 1


        if epoch % 10 == 0:

            val_classes = [
                1 if p > 0.5 else 0
                for p in val_probs
            ]

            val_accuracy = (
                accuracy_score(
                    val_labels,
                    val_classes
                )
            )

            print(
                f"Epoch {epoch:03d}/{NUM_EPOCHS} "
                f"| Train Loss: "
                f"{avg_train_loss:.4f} "
                f"| Val Loss: "
                f"{avg_val_loss:.4f} "
                f"| Val Acc: "
                f"{val_accuracy * 100:.2f}%"
            )


        if (
            patience_counter
            >= EARLY_STOPPING_PATIENCE
        ):

            print(
                f"Early stopping at "
                f"epoch {epoch}"
            )

            break


    # ========================================================
    # 13. RESTORE BEST MODEL
    # ========================================================

    if best_model_state is not None:

        rac_model.load_state_dict(
            best_model_state
        )

    rac_model.to(device)

    rac_model.eval()

    rac_model.gae_encoder.eval()


    # ========================================================
    # 14. FINAL VALIDATION EVALUATION
    # ========================================================

    final_labels = []
    final_probs = []
    final_subjects = []


    with torch.no_grad():

        for batch_idx, data in enumerate(
            val_loader
        ):

            data = data.to(device)


            v_query = (
                rac_model.gae_encoder(
                    data.x,
                    data.edge_index,
                    data.edge_weight,
                    data.batch
                )
            )


            query_np = (
                v_query
                .cpu()
                .numpy()
                .astype("float32")
            )


            _, neighbor_indices = (
                index.search(
                    query_np,
                    K
                )
            )


            v_retrieved = (
                torch.from_numpy(
                    kb_embeddings[
                        neighbor_indices
                    ]
                )
                .float()
                .to(device)
            )


            predictions, _ = rac_model(
                data.x,
                data.edge_index,
                data.edge_weight,
                data.batch,
                v_retrieved
            )


            probability = float(
                predictions
                .view(-1)[0]
                .cpu()
                .item()
            )

            label = int(
                data.y
                .view(-1)[0]
                .cpu()
                .item()
            )


            final_probs.append(
                probability
            )

            final_labels.append(
                label
            )

            final_subjects.append(
                df_val.iloc[
                    batch_idx
                ]["subject_id"]
            )


    final_classes = [
        1 if p > 0.5 else 0
        for p in final_probs
    ]


    # ========================================================
    # 15. METRICS
    # ========================================================

    accuracy = accuracy_score(
        final_labels,
        final_classes
    )

    precision = precision_score(
        final_labels,
        final_classes,
        zero_division=0
    )

    sensitivity = recall_score(
        final_labels,
        final_classes,
        pos_label=1,
        zero_division=0
    )

    f1 = f1_score(
        final_labels,
        final_classes,
        zero_division=0
    )


    try:

        roc_auc = roc_auc_score(
            final_labels,
            final_probs
        )

    except ValueError:

        roc_auc = np.nan


    cm = confusion_matrix(
        final_labels,
        final_classes,
        labels=[0, 1]
    )

    tn, fp, fn, tp = cm.ravel()


    specificity = (
        tn / (tn + fp)
        if (tn + fp) > 0
        else np.nan
    )


    print(
        f"\nBest epoch: {best_epoch}"
    )

    print(
        f"Accuracy: "
        f"{accuracy * 100:.2f}%"
    )

    print(
        f"ROC-AUC: "
        f"{roc_auc:.4f}"
    )

    print(
        f"Sensitivity: "
        f"{sensitivity:.4f}"
    )

    print(
        f"Specificity: "
        f"{specificity:.4f}"
    )

    print(
        "Confusion Matrix:"
    )

    print(cm)


    # ========================================================
    # 16. SAVE FOLD RESULT
    # ========================================================

    all_fold_results.append({

        "repeat":
            repeat_number,

        "fold":
            fold_in_repeat,

        "overall_fold_number":
            fold_number,

        "best_epoch":
            best_epoch,

        "best_val_loss":
            best_val_loss,

        "accuracy":
            accuracy,

        "accuracy_percent":
            accuracy * 100,

        "roc_auc":
            roc_auc,

        "precision":
            precision,

        "sensitivity":
            sensitivity,

        "specificity":
            specificity,

        "f1":
            f1,

        "tn":
            tn,

        "fp":
            fp,

        "fn":
            fn,

        "tp":
            tp

    })


    # Save subject-level predictions
    for (
        subject_id,
        label,
        probability,
        predicted_class
    ) in zip(
        final_subjects,
        final_labels,
        final_probs,
        final_classes
    ):

        all_predictions.append({

            "repeat":
                repeat_number,

            "fold":
                fold_in_repeat,

            "subject_id":
                subject_id,

            "ground_truth":
                label,

            "probability":
                probability,

            "predicted_class":
                predicted_class

        })


# ============================================================
# 17. CREATE RESULTS DATAFRAMES
# ============================================================

results_df = pd.DataFrame(
    all_fold_results
)

predictions_df = pd.DataFrame(
    all_predictions
)


# ============================================================
# 18. OVERALL SUMMARY
# ============================================================

summary = {

    "model":
        "RAC v2 Linear Attention",

    "cv_method":
        "Repeated Stratified 5-Fold CV",

    "n_splits":
        N_SPLITS,

    "n_repeats":
        N_REPEATS,

    "total_folds":
        len(results_df),

    "mean_accuracy_percent":
        results_df[
            "accuracy_percent"
        ].mean(),

    "std_accuracy_percent":
        results_df[
            "accuracy_percent"
        ].std(),

    "mean_roc_auc":
        results_df[
            "roc_auc"
        ].mean(),

    "std_roc_auc":
        results_df[
            "roc_auc"
        ].std(),

    "mean_precision":
        results_df[
            "precision"
        ].mean(),

    "mean_sensitivity":
        results_df[
            "sensitivity"
        ].mean(),

    "mean_specificity":
        results_df[
            "specificity"
        ].mean(),

    "mean_f1":
        results_df[
            "f1"
        ].mean(),

    "mean_best_epoch":
        results_df[
            "best_epoch"
        ].mean()

}

summary_df = pd.DataFrame(
    [summary]
)


# ============================================================
# 19. SAVE CSV FILES
# ============================================================

fold_results_path = os.path.join(
    results_dir,
    "sana_rac_v2_repeated_cv_folds.csv"
)

predictions_path = os.path.join(
    results_dir,
    "sana_rac_v2_repeated_cv_predictions.csv"
)

summary_path = os.path.join(
    results_dir,
    "sana_rac_v2_repeated_cv_summary.csv"
)


results_df.to_csv(
    fold_results_path,
    index=False
)

predictions_df.to_csv(
    predictions_path,
    index=False
)

summary_df.to_csv(
    summary_path,
    index=False
)


# ============================================================
# 20. PRINT FINAL SUMMARY
# ============================================================

print("\n")
print("=" * 60)

print(
    "RAC v2 REPEATED CROSS-VALIDATION SUMMARY"
)

print("=" * 60)

print(
    f"Total folds: "
    f"{len(results_df)}"
)

print(
    f"\nAccuracy: "
    f"{summary['mean_accuracy_percent']:.2f}% "
    f"± "
    f"{summary['std_accuracy_percent']:.2f}%"
)

print(
    f"ROC-AUC: "
    f"{summary['mean_roc_auc']:.4f} "
    f"± "
    f"{summary['std_roc_auc']:.4f}"
)

print(
    f"Precision: "
    f"{summary['mean_precision']:.4f}"
)

print(
    f"Sensitivity: "
    f"{summary['mean_sensitivity']:.4f}"
)

print(
    f"Specificity: "
    f"{summary['mean_specificity']:.4f}"
)

print(
    f"F1-score: "
    f"{summary['mean_f1']:.4f}"
)

print(
    f"Average best epoch: "
    f"{summary['mean_best_epoch']:.1f}"
)

print("\nFiles saved:")

print(fold_results_path)
print(predictions_path)
print(summary_path)