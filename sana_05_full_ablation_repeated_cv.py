# ============================================================
# SANA 05
# FULL RAC ABLATION WITH REPEATED STRATIFIED 5-FOLD CV
# ============================================================
#
# Experiments:
#
# E0 - Ab-initio
#      Fresh/trainable GNN encoder
#      No external pretraining
#      No retrieval
#      No attention
#
# E1 - Pretrained only
#      Frozen pretrained GAE encoder
#      No retrieval
#      No attention
#
# E2 - Retrieval mean
#      Frozen pretrained GAE encoder
#      FAISS retrieval K=10
#      Retrieved embeddings averaged equally
#      No attention
#
# E3 - RAC v1
#      Frozen pretrained GAE encoder
#      FAISS retrieval K=10
#      Dot-product attention
#
# E4 - RAC v2
#      Frozen pretrained GAE encoder
#      FAISS retrieval K=10
#      Learnable linear attention
#
# IMPORTANT:
# All five models use EXACTLY THE SAME CV splits.
#
# ============================================================


# ============================================================
# 1. IMPORTS
# ============================================================

import os
import copy
import random

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
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
    AttentionMechanism,
    AttentionMechanismLinear,
    RAC_Model,
    Baseline_GNN
)

from utils import FCDataset


# ============================================================
# 2. SETTINGS
# ============================================================

SEED = 42

N_SPLITS = 5

N_REPEATS = 5

K = 10

BATCH_SIZE = 16

MAX_EPOCHS = 100

LEARNING_RATE = 0.0001

WEIGHT_DECAY = 1e-4

EARLY_STOPPING_PATIENCE = 15


# ============================================================
# 3. REPRODUCIBILITY
# ============================================================

def set_seed(seed):

    random.seed(seed)

    np.random.seed(seed)

    torch.manual_seed(seed)

    if torch.cuda.is_available():

        torch.cuda.manual_seed_all(seed)


set_seed(SEED)


# ============================================================
# 4. DEVICE
# ============================================================

device = torch.device(
    "cuda"
    if torch.cuda.is_available()
    else "cpu"
)

print("\nUsing device:", device)


# ============================================================
# 5. PATHS
# ============================================================

output_dir = (
    r"C:\Users\imam\Desktop\Tara Project\processed_data_1"
)

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
# 6. LOAD DATA
# ============================================================

df = pd.read_csv(master_csv)


# ------------------------------------------------------------
# Fix Tara's absolute npy paths
# ------------------------------------------------------------

df["npy_path"] = df["npy_path"].apply(
    lambda x: os.path.join(
        output_dir,
        os.path.basename(x)
    )
)


# ------------------------------------------------------------
# TaoWu = target dataset
# ------------------------------------------------------------

df_target = df[
    df["dataset_source"] == "TaoWu"
].reset_index(drop=True)


print("\n========================================")
print("TAOWU DATA")
print("========================================")

print(
    df_target["label"]
    .value_counts()
    .sort_index()
)

print(
    "\nTotal TaoWu subjects:",
    len(df_target)
)


# ============================================================
# 7. LOAD FAISS KNOWLEDGE BASE
# ============================================================

index = faiss.read_index(
    index_path
)

kb_embeddings = np.load(
    kb_embeddings_path
).astype("float32")


print(
    "\nKnowledge-base embeddings:",
    kb_embeddings.shape
)


# ============================================================
# 8. HELPER: LOAD PRETRAINED ENCODER
# ============================================================

def create_pretrained_encoder():

    gae = GraphAutoencoder(
        num_nodes=100,
        input_dim=100,
        hidden_dim=64,
        embedding_dim=128
    )

    state = torch.load(
        encoder_weights,
        map_location=device
    )

    gae.encoder.load_state_dict(
        state
    )

    gae.encoder.to(device)

    # Freeze pretrained encoder
    for parameter in gae.encoder.parameters():

        parameter.requires_grad = False

    gae.encoder.eval()

    return gae.encoder


# ============================================================
# 9. E1 MODEL:
# PRETRAINED ONLY
# ============================================================

class PretrainedOnlyClassifier(nn.Module):

    def __init__(self):

        super().__init__()

        self.gae_encoder = (
            create_pretrained_encoder()
        )

        self.classification_head = nn.Sequential(

            nn.Linear(
                128,
                64
            ),

            nn.ReLU(),

            nn.Dropout(
                0.3
            ),

            nn.Linear(
                64,
                1
            ),

            nn.Sigmoid()
        )


    def forward(
        self,
        x,
        edge_index,
        edge_weight,
        batch
    ):

        embedding = self.gae_encoder(
            x,
            edge_index,
            edge_weight,
            batch
        )

        prediction = (
            self.classification_head(
                embedding
            )
        )

        return prediction


# ============================================================
# 10. E2 MODEL:
# RETRIEVAL + SIMPLE MEAN
# ============================================================

class RetrievalMeanClassifier(nn.Module):

    def __init__(self):

        super().__init__()

        self.gae_encoder = (
            create_pretrained_encoder()
        )

        self.classification_head = nn.Sequential(

            nn.Linear(
                256,
                64
            ),

            nn.ReLU(),

            nn.Dropout(
                0.3
            ),

            nn.Linear(
                64,
                1
            ),

            nn.Sigmoid()
        )


    def forward(
        self,
        x,
        edge_index,
        edge_weight,
        batch,
        retrieved_embeddings
    ):

        query_embedding = (
            self.gae_encoder(
                x,
                edge_index,
                edge_weight,
                batch
            )
        )

        # Equal weighting of all K retrieved neighbors
        context_embedding = (
            retrieved_embeddings.mean(
                dim=1
            )
        )

        augmented_embedding = torch.cat(
            (
                query_embedding,
                context_embedding
            ),
            dim=1
        )

        prediction = (
            self.classification_head(
                augmented_embedding
            )
        )

        return prediction


# ============================================================
# 11. CREATE MODEL
# ============================================================

def create_model(
    experiment
):

    # --------------------------------------------------------
    # E0 - AB INITIO
    # --------------------------------------------------------

    if experiment == "E0 - Ab-initio":

        fresh_gae = GraphAutoencoder(
            num_nodes=100,
            input_dim=100,
            hidden_dim=64,
            embedding_dim=128
        )

        model = Baseline_GNN(
            gae_encoder=fresh_gae.encoder,
            embedding_dim=128
        )


    # --------------------------------------------------------
    # E1 - PRETRAINED ONLY
    # --------------------------------------------------------

    elif experiment == "E1 - Pretrained only":

        model = (
            PretrainedOnlyClassifier()
        )


    # --------------------------------------------------------
    # E2 - RETRIEVAL MEAN
    # --------------------------------------------------------

    elif experiment == "E2 - Retrieval mean":

        model = (
            RetrievalMeanClassifier()
        )


    # --------------------------------------------------------
    # E3 - RAC V1
    # --------------------------------------------------------

    elif experiment == "E3 - RAC v1":

        pretrained_encoder = (
            create_pretrained_encoder()
        )

        attention = (
            AttentionMechanism(
                embedding_dim=128
            )
        )

        model = RAC_Model(
            gae_encoder=pretrained_encoder,
            attention_model=attention,
            embedding_dim=128
        )


    # --------------------------------------------------------
    # E4 - RAC V2
    # --------------------------------------------------------

    elif experiment == "E4 - RAC v2":

        pretrained_encoder = (
            create_pretrained_encoder()
        )

        attention = (
            AttentionMechanismLinear(
                embedding_dim=128
            )
        )

        model = RAC_Model(
            gae_encoder=pretrained_encoder,
            attention_model=attention,
            embedding_dim=128
        )


    else:

        raise ValueError(
            "Unknown experiment: "
            + experiment
        )


    return model.to(device)


# ============================================================
# 12. RETRIEVE KNOWLEDGE-BASE NEIGHBORS
# ============================================================

def retrieve_neighbors(
    encoder,
    data
):

    # Frozen encoder should stay deterministic
    encoder.eval()

    with torch.no_grad():

        query_embedding = encoder(
            data.x,
            data.edge_index,
            data.edge_weight,
            data.batch
        )


    query_np = (
        query_embedding
        .detach()
        .cpu()
        .numpy()
        .astype("float32")
    )


    _, neighbor_indices = index.search(
        query_np,
        K
    )


    retrieved = torch.from_numpy(
        kb_embeddings[
            neighbor_indices
        ]
    ).float().to(device)


    return retrieved


# ============================================================
# 13. MODEL FORWARD FUNCTION
# ============================================================

def forward_model(
    experiment,
    model,
    data
):

    # --------------------------------------------------------
    # E0
    # --------------------------------------------------------

    if experiment == "E0 - Ab-initio":

        return model(
            data.x,
            data.edge_index,
            data.edge_weight,
            data.batch
        )


    # --------------------------------------------------------
    # E1
    # --------------------------------------------------------

    if experiment == "E1 - Pretrained only":

        return model(
            data.x,
            data.edge_index,
            data.edge_weight,
            data.batch
        )


    # --------------------------------------------------------
    # E2
    # --------------------------------------------------------

    if experiment == "E2 - Retrieval mean":

        retrieved = retrieve_neighbors(
            model.gae_encoder,
            data
        )

        return model(
            data.x,
            data.edge_index,
            data.edge_weight,
            data.batch,
            retrieved
        )


    # --------------------------------------------------------
    # E3 / E4
    # --------------------------------------------------------

    if experiment in [
        "E3 - RAC v1",
        "E4 - RAC v2"
    ]:

        retrieved = retrieve_neighbors(
            model.gae_encoder,
            data
        )

        prediction, _ = model(
            data.x,
            data.edge_index,
            data.edge_weight,
            data.batch,
            retrieved
        )

        return prediction


# ============================================================
# 14. KEEP FROZEN ENCODER IN EVAL MODE
# ============================================================

def set_correct_training_mode(
    experiment,
    model
):

    model.train()


    # E0 encoder is NOT frozen.
    # It must remain in training mode.

    if experiment != "E0 - Ab-initio":

        model.gae_encoder.eval()


# ============================================================
# 15. VALIDATION FUNCTION
# ============================================================

def validate_model(
    experiment,
    model,
    val_loader,
    loss_fn
):

    model.eval()

    # Frozen encoder explicitly stays eval
    if experiment != "E0 - Ab-initio":

        model.gae_encoder.eval()


    labels_list = []

    probabilities_list = []

    total_loss = 0.0


    with torch.no_grad():

        for data in val_loader:

            data = data.to(device)


            predictions = forward_model(
                experiment,
                model,
                data
            )


            predictions = (
                predictions.view(-1)
            )

            labels = (
                data.y
                .float()
                .view(-1)
            )


            loss = loss_fn(
                predictions,
                labels
            )


            total_loss += (
                loss.item()
            )


            labels_list.extend(
                labels
                .cpu()
                .numpy()
                .tolist()
            )


            probabilities_list.extend(
                predictions
                .cpu()
                .numpy()
                .tolist()
            )


    average_loss = (
        total_loss /
        len(val_loader)
    )


    return (
        average_loss,
        labels_list,
        probabilities_list
    )


# ============================================================
# 16. CALCULATE METRICS
# ============================================================

def calculate_metrics(
    labels,
    probabilities
):

    predicted_classes = [

        1
        if probability > 0.5
        else 0

        for probability
        in probabilities
    ]


    accuracy = accuracy_score(
        labels,
        predicted_classes
    )


    precision = precision_score(
        labels,
        predicted_classes,
        zero_division=0
    )


    sensitivity = recall_score(
        labels,
        predicted_classes,
        pos_label=1,
        zero_division=0
    )


    f1 = f1_score(
        labels,
        predicted_classes,
        zero_division=0
    )


    try:

        roc_auc = roc_auc_score(
            labels,
            probabilities
        )

    except ValueError:

        roc_auc = np.nan


    cm = confusion_matrix(
        labels,
        predicted_classes,
        labels=[0, 1]
    )


    tn, fp, fn, tp = (
        cm.ravel()
    )


    specificity = (

        tn / (tn + fp)

        if (tn + fp) > 0

        else np.nan
    )


    return {

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
            tp,

        "predicted_classes":
            predicted_classes
    }


# ============================================================
# 17. CREATE SAME CV SPLITS FOR ALL MODELS
# ============================================================

cv = RepeatedStratifiedKFold(
    n_splits=N_SPLITS,
    n_repeats=N_REPEATS,
    random_state=SEED
)


splits = list(
    cv.split(
        df_target,
        df_target["label"]
    )
)


print("\n========================================")
print("CROSS-VALIDATION")
print("========================================")

print(
    "Splits:",
    N_SPLITS
)

print(
    "Repeats:",
    N_REPEATS
)

print(
    "Total folds per model:",
    len(splits)
)


# ============================================================
# 18. EXPERIMENTS
# ============================================================

experiments = [

    "E0 - Ab-initio",

    "E1 - Pretrained only",

    "E2 - Retrieval mean",

    "E3 - RAC v1",

    "E4 - RAC v2"

]


# ============================================================
# 19. STORAGE
# ============================================================

all_results = []

all_predictions = []


# ============================================================
# 20. MAIN EXPERIMENT LOOP
# ============================================================

for experiment_number, experiment in enumerate(
    experiments,
    start=1
):

    print("\n")
    print("#" * 70)

    print(
        f"EXPERIMENT {experiment_number}/"
        f"{len(experiments)}"
    )

    print(experiment)

    print("#" * 70)


    for split_number, (
        train_idx,
        val_idx
    ) in enumerate(
        splits,
        start=1
    ):


        repeat_number = (
            (split_number - 1)
            // N_SPLITS
        ) + 1


        fold_number = (
            (split_number - 1)
            % N_SPLITS
        ) + 1


        print("\n" + "=" * 60)

        print(
            f"{experiment}"
        )

        print(
            f"Repeat "
            f"{repeat_number}/"
            f"{N_REPEATS}"
            f" | Fold "
            f"{fold_number}/"
            f"{N_SPLITS}"
        )

        print("=" * 60)


        # ====================================================
        # 21. REPRODUCIBLE MODEL INITIALIZATION
        # ====================================================

        # Same deterministic seed for corresponding split
        fold_seed = (
            SEED
            + split_number
        )

        set_seed(
            fold_seed
        )


        # ====================================================
        # 22. TRAIN / VALIDATION DATA
        # ====================================================

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


        train_dataset = (
            FCDataset(
                df_train
            )
        )

        val_dataset = (
            FCDataset(
                df_val
            )
        )


        # Separate seeded generator
        generator = (
            torch.Generator()
        )

        generator.manual_seed(
            fold_seed
        )


        train_loader = DataLoader(

            train_dataset,

            batch_size=BATCH_SIZE,

            shuffle=True,

            generator=generator

        )


        # Batch size 1 lets us save
        # individual predictions cleanly
        val_loader = DataLoader(

            val_dataset,

            batch_size=1,

            shuffle=False

        )


        # ====================================================
        # 23. CREATE FRESH MODEL
        # ====================================================

        model = create_model(
            experiment
        )


        # ====================================================
        # 24. OPTIMIZER
        # ====================================================

        trainable_parameters = [

            parameter

            for parameter
            in model.parameters()

            if parameter.requires_grad

        ]


        optimizer = torch.optim.Adam(

            trainable_parameters,

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


        loss_fn = (
            torch.nn.BCELoss()
        )


        # ====================================================
        # 25. EARLY STOPPING
        # ====================================================

        best_val_loss = (
            float("inf")
        )

        best_epoch = 0

        best_state = None

        patience_counter = 0


        # ====================================================
        # 26. TRAINING LOOP
        # ====================================================

        for epoch in range(
            1,
            MAX_EPOCHS + 1
        ):


            set_correct_training_mode(
                experiment,
                model
            )


            total_train_loss = (
                0.0
            )


            for data in train_loader:

                data = data.to(
                    device
                )


                optimizer.zero_grad()


                predictions = (
                    forward_model(
                        experiment,
                        model,
                        data
                    )
                )


                predictions = (
                    predictions
                    .view(-1)
                )


                labels = (
                    data.y
                    .float()
                    .view(-1)
                )


                loss = loss_fn(
                    predictions,
                    labels
                )


                loss.backward()


                torch.nn.utils.clip_grad_norm_(

                    trainable_parameters,

                    max_norm=1.0

                )


                optimizer.step()


                total_train_loss += (
                    loss.item()
                )


            average_train_loss = (

                total_train_loss /
                len(train_loader)

            )


            # =================================================
            # VALIDATION
            # =================================================

            (
                val_loss,
                val_labels,
                val_probabilities

            ) = validate_model(

                experiment,

                model,

                val_loader,

                loss_fn
            )


            scheduler.step(
                val_loss
            )


            # =================================================
            # SAVE LOWEST VALIDATION LOSS MODEL
            # =================================================

            if val_loss < best_val_loss:

                best_val_loss = (
                    val_loss
                )

                best_epoch = (
                    epoch
                )

                patience_counter = (
                    0
                )


                best_state = {

                    key:
                        value
                        .detach()
                        .cpu()
                        .clone()

                    for key, value
                    in model
                    .state_dict()
                    .items()

                }


            else:

                patience_counter += (
                    1
                )


            # =================================================
            # PRINT EVERY 10 EPOCHS
            # =================================================

            if epoch % 10 == 0:

                current_metrics = (
                    calculate_metrics(

                        val_labels,

                        val_probabilities

                    )
                )


                print(

                    f"Epoch "
                    f"{epoch:03d}/"
                    f"{MAX_EPOCHS}"

                    f" | Train Loss: "
                    f"{average_train_loss:.4f}"

                    f" | Val Loss: "
                    f"{val_loss:.4f}"

                    f" | Val Acc: "
                    f"{current_metrics['accuracy_percent']:.2f}%"

                )


            # =================================================
            # EARLY STOP
            # =================================================

            if (
                patience_counter
                >= EARLY_STOPPING_PATIENCE
            ):

                print(

                    f"Early stopping "
                    f"at epoch "
                    f"{epoch}"

                )

                break


        # ====================================================
        # 27. RESTORE BEST CHECKPOINT
        # ====================================================

        if best_state is not None:

            model.load_state_dict(
                best_state
            )


        model.to(
            device
        )

        model.eval()


        if experiment != "E0 - Ab-initio":

            model.gae_encoder.eval()


        # ====================================================
        # 28. FINAL EVALUATION
        # ====================================================

        final_labels = []

        final_probabilities = []

        final_subject_ids = []


        with torch.no_grad():

            for row_number, data in enumerate(
                val_loader
            ):

                data = data.to(
                    device
                )


                predictions = (
                    forward_model(

                        experiment,

                        model,

                        data

                    )
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


                final_probabilities.append(
                    probability
                )

                final_labels.append(
                    label
                )


                final_subject_ids.append(

                    df_val
                    .iloc[row_number]
                    ["subject_id"]

                )


        # ====================================================
        # 29. FINAL METRICS
        # ====================================================

        metrics = calculate_metrics(

            final_labels,

            final_probabilities

        )


        print(
            f"\nBest epoch: "
            f"{best_epoch}"
        )

        print(
            f"Accuracy: "
            f"{metrics['accuracy_percent']:.2f}%"
        )

        print(
            f"ROC-AUC: "
            f"{metrics['roc_auc']:.4f}"
        )

        print(
            f"Sensitivity: "
            f"{metrics['sensitivity']:.4f}"
        )

        print(
            f"Specificity: "
            f"{metrics['specificity']:.4f}"
        )

        print(
            "Confusion Matrix:"
        )

        print(
            [
                [
                    metrics["tn"],
                    metrics["fp"]
                ],
                [
                    metrics["fn"],
                    metrics["tp"]
                ]
            ]
        )


        # ====================================================
        # 30. SAVE FOLD RESULT
        # ====================================================

        all_results.append({

            "experiment":
                experiment,

            "repeat":
                repeat_number,

            "fold":
                fold_number,

            "split_number":
                split_number,

            "best_epoch":
                best_epoch,

            "best_val_loss":
                best_val_loss,

            "accuracy":
                metrics["accuracy"],

            "accuracy_percent":
                metrics[
                    "accuracy_percent"
                ],

            "roc_auc":
                metrics["roc_auc"],

            "precision":
                metrics["precision"],

            "sensitivity":
                metrics["sensitivity"],

            "specificity":
                metrics["specificity"],

            "f1":
                metrics["f1"],

            "tn":
                metrics["tn"],

            "fp":
                metrics["fp"],

            "fn":
                metrics["fn"],

            "tp":
                metrics["tp"]

        })


        # ====================================================
        # 31. SAVE SUBJECT PREDICTIONS
        # ====================================================

        for (
            subject_id,
            label,
            probability,
            predicted_class

        ) in zip(

            final_subject_ids,

            final_labels,

            final_probabilities,

            metrics[
                "predicted_classes"
            ]

        ):

            all_predictions.append({

                "experiment":
                    experiment,

                "repeat":
                    repeat_number,

                "fold":
                    fold_number,

                "split_number":
                    split_number,

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
# 32. RAW RESULTS DATAFRAME
# ============================================================

results_df = pd.DataFrame(
    all_results
)


predictions_df = pd.DataFrame(
    all_predictions
)


# ============================================================
# 33. SUMMARY TABLE
# ============================================================

summary_df = (

    results_df

    .groupby(
        "experiment",
        sort=False
    )

    .agg(

        mean_accuracy_percent=(
            "accuracy_percent",
            "mean"
        ),

        std_accuracy_percent=(
            "accuracy_percent",
            "std"
        ),

        mean_roc_auc=(
            "roc_auc",
            "mean"
        ),

        std_roc_auc=(
            "roc_auc",
            "std"
        ),

        mean_precision=(
            "precision",
            "mean"
        ),

        mean_sensitivity=(
            "sensitivity",
            "mean"
        ),

        mean_specificity=(
            "specificity",
            "mean"
        ),

        mean_f1=(
            "f1",
            "mean"
        ),

        mean_best_epoch=(
            "best_epoch",
            "mean"
        )

    )

    .reset_index()

)


# ============================================================
# 34. COMPONENT CONTRIBUTIONS
# ============================================================

accuracy_lookup = dict(

    zip(

        summary_df[
            "experiment"
        ],

        summary_df[
            "mean_accuracy_percent"
        ]

    )
)


contribution_rows = []


def add_comparison(
    component,
    from_experiment,
    to_experiment
):

    from_accuracy = (
        accuracy_lookup[
            from_experiment
        ]
    )

    to_accuracy = (
        accuracy_lookup[
            to_experiment
        ]
    )


    contribution_rows.append({

        "component":
            component,

        "from_experiment":
            from_experiment,

        "to_experiment":
            to_experiment,

        "from_accuracy_percent":
            from_accuracy,

        "to_accuracy_percent":
            to_accuracy,

        "difference_percentage_points":
            to_accuracy
            - from_accuracy

    })


# Pretraining contribution
add_comparison(

    "External pretraining",

    "E0 - Ab-initio",

    "E1 - Pretrained only"

)


# Retrieval contribution
add_comparison(

    "Mean retrieval",

    "E1 - Pretrained only",

    "E2 - Retrieval mean"

)


# Dot-product attention contribution
add_comparison(

    "Dot-product attention",

    "E2 - Retrieval mean",

    "E3 - RAC v1"

)


# Linear attention contribution
add_comparison(

    "Linear attention",

    "E2 - Retrieval mean",

    "E4 - RAC v2"

)


contribution_df = pd.DataFrame(
    contribution_rows
)


# ============================================================
# 35. SAVE FILES
# ============================================================

raw_results_path = os.path.join(

    results_dir,

    "sana_full_ablation_all_folds.csv"

)


predictions_path = os.path.join(

    results_dir,

    "sana_full_ablation_predictions.csv"

)


summary_path = os.path.join(

    results_dir,

    "sana_full_ablation_summary.csv"

)


contribution_path = os.path.join(

    results_dir,

    "sana_full_ablation_component_changes.csv"

)


results_df.to_csv(

    raw_results_path,

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


contribution_df.to_csv(

    contribution_path,

    index=False

)


# ============================================================
# 36. PRINT FINAL RESULTS
# ============================================================

print("\n\n")
print("=" * 80)

print(
    "FINAL FULL ABLATION SUMMARY"
)

print("=" * 80)


print(

    summary_df[
        [
            "experiment",
            "mean_accuracy_percent",
            "std_accuracy_percent",
            "mean_roc_auc",
            "std_roc_auc",
            "mean_sensitivity",
            "mean_specificity",
            "mean_f1"
        ]
    ].to_string(
        index=False
    )

)


print("\n")
print("-" * 80)

print(
    "COMPONENT CHANGES IN ACCURACY"
)

print("-" * 80)


print(

    contribution_df.to_string(
        index=False
    )

)


print("\n")
print("=" * 80)

print("FILES SAVED")

print("=" * 80)


print(
    raw_results_path
)

print(
    predictions_path
)

print(
    summary_path
)

print(
    contribution_path
)