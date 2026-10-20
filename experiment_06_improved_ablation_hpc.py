# ============================================================
# SANA 06
# IMPROVED RAC ABLATION - HPC VERSION
# ============================================================
#
# Improvements tested:
#
# 1. Different retrieval sizes:
#       K = 5, 10, 15
#
# 2. Limited fine-tuning:
#       - conv1 frozen
#       - conv2 frozen
#       - lin_encode trainable
#
# Evaluation:
#       Repeated Stratified 5-Fold CV
#       5 repeats = 25 folds per experiment
#
# IMPORTANT:
#       Tara's original files/models are NOT overwritten.
#
# ============================================================


# ============================================================
# 1. IMPORTS
# ============================================================

import os
import copy
import random
from pathlib import Path

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
# 2. GENERAL SETTINGS
# ============================================================

SEED = 42

N_SPLITS = 5
N_REPEATS = 5

BATCH_SIZE = 16
MAX_EPOCHS = 100

LEARNING_RATE = 1e-4
WEIGHT_DECAY = 1e-4

EARLY_STOPPING_PATIENCE = 15

# Retrieval values to test
K_VALUES = [5, 10, 15]


# ============================================================
# 3. HPC PATHS
# ============================================================

PROJECT_ROOT = Path(
    "/fs/dss/work/nuji0350/tara-project"
)

REPO_DIR = (
    PROJECT_ROOT /
    "practical-project"
)

# ------------------------------------------------------------
# Automatically find processed_data_1
# ------------------------------------------------------------

possible_processed_dirs = [

    PROJECT_ROOT /
    "processed_data_1",

    REPO_DIR /
    "processed_data_1",
]


PROCESSED_DIR = None

for candidate in possible_processed_dirs:

    if (
        candidate.exists()
        and
        (candidate / "master_metadata.csv").exists()
    ):

        PROCESSED_DIR = candidate
        break


if PROCESSED_DIR is None:

    raise FileNotFoundError(
        "Could not find processed_data_1.\n"
        "Checked:\n"
        + "\n".join(
            str(x)
            for x in possible_processed_dirs
        )
    )


print(
    "\nUsing processed data directory:"
)

print(
    PROCESSED_DIR
)


# ============================================================
# 4. REQUIRED FILE PATHS
# ============================================================

MASTER_CSV = (
    PROCESSED_DIR /
    "master_metadata.csv"
)

ENCODER_WEIGHTS = (
    PROCESSED_DIR /
    "gae_encoder.pth"
)

INDEX_PATH = (
    PROCESSED_DIR /
    "knowledge_base.index"
)

KB_EMBEDDINGS_PATH = (
    PROCESSED_DIR /
    "kb_embeddings.npy"
)


RESULTS_DIR = (
    PROCESSED_DIR /
    "sana_results"
)

RESULTS_DIR.mkdir(
    parents=True,
    exist_ok=True
)


# ============================================================
# 5. CHECK REQUIRED FILES
# ============================================================

required_files = [

    MASTER_CSV,
    ENCODER_WEIGHTS,
    INDEX_PATH,
    KB_EMBEDDINGS_PATH
]


for file_path in required_files:

    if not file_path.exists():

        raise FileNotFoundError(
            f"Missing required file:\n"
            f"{file_path}"
        )


# ============================================================
# 6. REPRODUCIBILITY
# ============================================================

def set_seed(seed):

    random.seed(seed)

    np.random.seed(seed)

    torch.manual_seed(seed)

    if torch.cuda.is_available():

        torch.cuda.manual_seed_all(seed)


set_seed(SEED)


# ============================================================
# 7. DEVICE
# ============================================================

device = torch.device(
    "cuda"
    if torch.cuda.is_available()
    else "cpu"
)


print(
    "\nUsing device:",
    device
)


# ============================================================
# 8. LOAD METADATA
# ============================================================

df = pd.read_csv(
    MASTER_CSV
)


# ============================================================
# 9. FIX WINDOWS PATHS FOR HPC
# ============================================================

# Tara's CSV may contain paths such as:
#
# C:\Users\nikna\Documents\...\patient001_fc_matrix.npy
#
# On HPC we only need the filename.

def fix_npy_path(old_path):

    old_path = str(
        old_path
    )

    # Handle Windows path separators
    filename = (
        old_path
        .replace("\\", "/")
        .split("/")[-1]
    )


    # First try main processed directory
    candidate = (
        PROCESSED_DIR /
        filename
    )


    if candidate.exists():

        return str(
            candidate
        )


    # Try other known processed_data folders
    for folder in possible_processed_dirs:

        candidate = (
            folder /
            filename
        )

        if candidate.exists():

            return str(
                candidate
            )


    # Return expected path so error is informative
    return str(
        PROCESSED_DIR /
        filename
    )


df["npy_path"] = (
    df["npy_path"]
    .apply(
        fix_npy_path
    )
)


# ============================================================
# 10. FILTER TAOWU
# ============================================================

df_target = df[
    df["dataset_source"]
    == "TaoWu"
].reset_index(
    drop=True
)


print("\n" + "=" * 60)

print(
    "TAOWU TARGET DATA"
)

print("=" * 60)


print(
    df_target[
        "label"
    ]
    .value_counts()
    .sort_index()
)


print(
    "\nTotal TaoWu subjects:",
    len(df_target)
)


# ============================================================
# 11. VERIFY ALL TAOWU FILES EXIST
# ============================================================

missing_files = []

for path in df_target["npy_path"]:

    if not os.path.exists(path):

        missing_files.append(
            path
        )


if missing_files:

    print(
        "\nMissing TaoWu files:"
    )

    for path in missing_files[:10]:

        print(
            path
        )

    raise FileNotFoundError(
        f"{len(missing_files)} TaoWu "
        f"FC files are missing."
    )


print(
    "\nAll TaoWu FC files found."
)


# ============================================================
# 12. LOAD FAISS / KNOWLEDGE BASE
# ============================================================

index = faiss.read_index(
    str(
        INDEX_PATH
    )
)


kb_embeddings = np.load(
    KB_EMBEDDINGS_PATH
).astype(
    "float32"
)


print(
    "\nKnowledge base shape:",
    kb_embeddings.shape
)


# ============================================================
# 13. PRETRAINED ENCODER
# ============================================================

def create_pretrained_encoder(
    fine_tune_last_layer=False
):

    gae = GraphAutoencoder(

        num_nodes=100,

        input_dim=100,

        hidden_dim=64,

        embedding_dim=128
    )


    state_dict = torch.load(

        ENCODER_WEIGHTS,

        map_location=device
    )


    gae.encoder.load_state_dict(
        state_dict
    )


    gae.encoder.to(
        device
    )


    # --------------------------------------------------------
    # Freeze everything first
    # --------------------------------------------------------

    for parameter in (
        gae.encoder.parameters()
    ):

        parameter.requires_grad = False


    # --------------------------------------------------------
    # LIMITED FINE-TUNING
    #
    # Only final embedding layer is trainable
    # --------------------------------------------------------

    if fine_tune_last_layer:

        for parameter in (
            gae.encoder
            .lin_encode
            .parameters()
        ):

            parameter.requires_grad = True


    gae.encoder.eval()


    return gae.encoder


# ============================================================
# 14. PRETRAINED-ONLY CLASSIFIER
# ============================================================

class PretrainedOnlyClassifier(
    nn.Module
):

    def __init__(self):

        super().__init__()


        self.gae_encoder = (
            create_pretrained_encoder(
                fine_tune_last_layer=False
            )
        )


        self.classification_head = (
            nn.Sequential(

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
        )


    def forward(
        self,
        x,
        edge_index,
        edge_weight,
        batch
    ):

        embedding = (
            self.gae_encoder(

                x,

                edge_index,

                edge_weight,

                batch
            )
        )


        return (
            self.classification_head(
                embedding
            )
        )


# ============================================================
# 15. RETRIEVAL-MEAN CLASSIFIER
# ============================================================

class RetrievalMeanClassifier(
    nn.Module
):

    def __init__(self):

        super().__init__()


        self.gae_encoder = (
            create_pretrained_encoder(
                fine_tune_last_layer=False
            )
        )


        self.classification_head = (
            nn.Sequential(

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
        )


    def forward(
        self,
        x,
        edge_index,
        edge_weight,
        batch,
        retrieved_embeddings
    ):


        query = (
            self.gae_encoder(

                x,

                edge_index,

                edge_weight,

                batch
            )
        )


        context = (
            retrieved_embeddings
            .mean(
                dim=1
            )
        )


        augmented = torch.cat(

            (
                query,
                context
            ),

            dim=1
        )


        return (
            self.classification_head(
                augmented
            )
        )


# ============================================================
# 16. CREATE MODEL
# ============================================================

def create_model(
    experiment
):


    # --------------------------------------------------------
    # E0 - AB INITIO
    # --------------------------------------------------------

    if experiment == "E0 - Ab-initio":

        fresh_gae = (
            GraphAutoencoder(

                num_nodes=100,

                input_dim=100,

                hidden_dim=64,

                embedding_dim=128
            )
        )


        model = Baseline_GNN(

            gae_encoder=
                fresh_gae.encoder,

            embedding_dim=128
        )


    # --------------------------------------------------------
    # E1 - PRETRAINED ONLY
    # --------------------------------------------------------

    elif experiment == (
        "E1 - Pretrained only"
    ):

        model = (
            PretrainedOnlyClassifier()
        )


    # --------------------------------------------------------
    # E2 - RETRIEVAL MEAN
    # --------------------------------------------------------

    elif experiment.startswith(
        "E2 - Retrieval mean"
    ):

        model = (
            RetrievalMeanClassifier()
        )


    # --------------------------------------------------------
    # E3 - DOT-PRODUCT ATTENTION
    # --------------------------------------------------------

    elif experiment.startswith(
        "E3 - RAC v1"
    ):

        encoder = (
            create_pretrained_encoder(
                False
            )
        )


        attention = (
            AttentionMechanism(
                embedding_dim=128
            )
        )


        model = RAC_Model(

            gae_encoder=encoder,

            attention_model=
                attention,

            embedding_dim=128
        )


    # --------------------------------------------------------
    # E4 - LINEAR ATTENTION, FROZEN ENCODER
    # --------------------------------------------------------

    elif experiment.startswith(
        "E4 - RAC v2 frozen"
    ):

        encoder = (
            create_pretrained_encoder(
                False
            )
        )


        attention = (
            AttentionMechanismLinear(
                embedding_dim=128
            )
        )


        model = RAC_Model(

            gae_encoder=encoder,

            attention_model=
                attention,

            embedding_dim=128
        )


    # --------------------------------------------------------
    # E5 - RAC V2 + LIMITED FINE-TUNING
    # --------------------------------------------------------

    elif experiment.startswith(
        "E5 - RAC v2 fine-tuned"
    ):

        encoder = (
            create_pretrained_encoder(
                fine_tune_last_layer=True
            )
        )


        attention = (
            AttentionMechanismLinear(
                embedding_dim=128
            )
        )


        model = RAC_Model(

            gae_encoder=encoder,

            attention_model=
                attention,

            embedding_dim=128
        )


    else:

        raise ValueError(
            f"Unknown experiment: "
            f"{experiment}"
        )


    return model.to(
        device
    )


# ============================================================
# 17. GET K FROM EXPERIMENT NAME
# ============================================================

def get_k(
    experiment
):

    if "K=" not in experiment:

        return None


    k_text = (
        experiment
        .split("K=")[-1]
        .split(")")[0]
        .strip()
    )


    return int(
        k_text
    )


# ============================================================
# 18. RETRIEVE NEIGHBOURS
# ============================================================

def retrieve_neighbors(
    encoder,
    data,
    k
):


    # --------------------------------------------------------
    # Important:
    # If encoder is fully frozen, no gradients needed.
    #
    # Even for fine-tuning, retrieval itself remains based on
    # detached query values because FAISS is non-differentiable.
    # --------------------------------------------------------

    query = encoder(

        data.x,

        data.edge_index,

        data.edge_weight,

        data.batch
    )


    query_np = (

        query
        .detach()
        .cpu()
        .numpy()
        .astype(
            "float32"
        )
    )


    _, neighbour_indices = (
        index.search(

            query_np,

            k
        )
    )


    retrieved = (

        torch.from_numpy(

            kb_embeddings[
                neighbour_indices
            ]

        )

        .float()

        .to(
            device
        )
    )


    return retrieved


# ============================================================
# 19. MODEL FORWARD
# ============================================================

def forward_model(
    experiment,
    model,
    data
):


    # --------------------------------------------------------
    # E0 / E1
    # --------------------------------------------------------

    if experiment in [

        "E0 - Ab-initio",

        "E1 - Pretrained only"
    ]:

        return model(

            data.x,

            data.edge_index,

            data.edge_weight,

            data.batch
        )


    # --------------------------------------------------------
    # Retrieval models
    # --------------------------------------------------------

    k = get_k(
        experiment
    )


    retrieved = (
        retrieve_neighbors(

            model.gae_encoder,

            data,

            k
        )
    )


    # --------------------------------------------------------
    # E2
    # --------------------------------------------------------

    if experiment.startswith(
        "E2 - Retrieval mean"
    ):

        return model(

            data.x,

            data.edge_index,

            data.edge_weight,

            data.batch,

            retrieved
        )


    # --------------------------------------------------------
    # E3/E4/E5
    # --------------------------------------------------------

    prediction, _ = model(

        data.x,

        data.edge_index,

        data.edge_weight,

        data.batch,

        retrieved
    )


    return prediction


# ============================================================
# 20. TRAINING MODE
# ============================================================

def set_training_mode(
    experiment,
    model
):


    model.train()


    # --------------------------------------------------------
    # E0:
    # train complete fresh encoder
    # --------------------------------------------------------

    if experiment == (
        "E0 - Ab-initio"
    ):

        return


    # --------------------------------------------------------
    # E5:
    # limited fine-tuning
    #
    # Keep convolution layers deterministic,
    # but train lin_encode
    # --------------------------------------------------------

    if experiment.startswith(
        "E5 - RAC v2 fine-tuned"
    ):

        model.gae_encoder.eval()

        # lin_encode itself has no dropout,
        # but remains trainable through gradients

        return


    # --------------------------------------------------------
    # All other pretrained encoders:
    # frozen and eval
    # --------------------------------------------------------

    model.gae_encoder.eval()


# ============================================================
# 21. VALIDATE
# ============================================================

def validate_model(
    experiment,
    model,
    loader,
    loss_fn
):


    model.eval()


    labels_list = []

    probabilities_list = []

    total_loss = 0.0


    with torch.no_grad():

        for data in loader:


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


    return (

        total_loss /
        len(loader),

        labels_list,

        probabilities_list
    )


# ============================================================
# 22. METRICS
# ============================================================

def calculate_metrics(
    labels,
    probabilities
):


    classes = [

        1
        if probability > 0.5
        else 0

        for probability
        in probabilities
    ]


    accuracy = accuracy_score(

        labels,

        classes
    )


    precision = precision_score(

        labels,

        classes,

        zero_division=0
    )


    sensitivity = recall_score(

        labels,

        classes,

        pos_label=1,

        zero_division=0
    )


    f1 = f1_score(

        labels,

        classes,

        zero_division=0
    )


    try:

        auc = roc_auc_score(

            labels,

            probabilities
        )

    except ValueError:

        auc = np.nan


    cm = confusion_matrix(

        labels,

        classes,

        labels=[0, 1]
    )


    tn, fp, fn, tp = (
        cm.ravel()
    )


    specificity = (

        tn /
        (tn + fp)

        if (
            tn + fp
        ) > 0

        else np.nan
    )


    return {

        "accuracy":
            accuracy,

        "accuracy_percent":
            accuracy * 100,

        "roc_auc":
            auc,

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

        "classes":
            classes
    }


# ============================================================
# 23. CROSS-VALIDATION SPLITS
# ============================================================

cv = RepeatedStratifiedKFold(

    n_splits=N_SPLITS,

    n_repeats=N_REPEATS,

    random_state=SEED
)


splits = list(

    cv.split(

        df_target,

        df_target[
            "label"
        ]
    )
)


print("\n" + "=" * 60)

print(
    "CROSS-VALIDATION SETUP"
)

print("=" * 60)


print(
    "Folds:",
    N_SPLITS
)

print(
    "Repeats:",
    N_REPEATS
)

print(
    "Validation folds per experiment:",
    len(splits)
)


# ============================================================
# 24. EXPERIMENT DEFINITIONS
# ============================================================

experiments = [

    "E0 - Ab-initio",

    "E1 - Pretrained only",
]


# ------------------------------------------------------------
# Test K for retrieval / attention models
# ------------------------------------------------------------

for k in K_VALUES:

    experiments.append(

        f"E2 - Retrieval mean (K={k})"
    )


for k in K_VALUES:

    experiments.append(

        f"E3 - RAC v1 (K={k})"
    )


for k in K_VALUES:

    experiments.append(

        f"E4 - RAC v2 frozen (K={k})"
    )


for k in K_VALUES:

    experiments.append(

        f"E5 - RAC v2 fine-tuned (K={k})"
    )


print(
    "\nExperiments:"
)

for experiment in experiments:

    print(
        "  ",
        experiment
    )


# ============================================================
# 25. STORAGE
# ============================================================

all_results = []

all_predictions = []


# ============================================================
# 26. MAIN LOOP
# ============================================================

for experiment_index, experiment in enumerate(

    experiments,

    start=1
):


    print("\n\n" + "#" * 75)

    print(

        f"EXPERIMENT "
        f"{experiment_index}/"
        f"{len(experiments)}"
    )

    print(
        experiment
    )

    print("#" * 75)


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


        print(
            "\n"
            + "=" * 60
        )


        print(

            f"{experiment} | "
            f"Repeat "
            f"{repeat_number}/"
            f"{N_REPEATS} | "
            f"Fold "
            f"{fold_number}/"
            f"{N_SPLITS}"
        )


        print(
            "=" * 60
        )


        # ----------------------------------------------------
        # deterministic seed per split
        # ----------------------------------------------------

        fold_seed = (

            SEED
            + split_number
        )


        set_seed(
            fold_seed
        )


        # ----------------------------------------------------
        # data split
        # ----------------------------------------------------

        df_train = (

            df_target
            .iloc[
                train_idx
            ]
            .reset_index(
                drop=True
            )
        )


        df_val = (

            df_target
            .iloc[
                val_idx
            ]
            .reset_index(
                drop=True
            )
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


        generator = (
            torch.Generator()
        )


        generator.manual_seed(
            fold_seed
        )


        train_loader = (
            DataLoader(

                train_dataset,

                batch_size=
                    BATCH_SIZE,

                shuffle=True,

                generator=
                    generator
            )
        )


        val_loader = (
            DataLoader(

                val_dataset,

                batch_size=1,

                shuffle=False
            )
        )


        # ----------------------------------------------------
        # model
        # ----------------------------------------------------

        model = (
            create_model(
                experiment
            )
        )


        trainable_parameters = [

            p

            for p in (
                model.parameters()
            )

            if p.requires_grad
        ]


        print(
            "Trainable parameters:",
            sum(
                p.numel()
                for p
                in trainable_parameters
            )
        )


        # ----------------------------------------------------
        # OPTIMIZER
        # ----------------------------------------------------

        # For limited fine tuning,
        # use smaller LR on pretrained lin_encode.

        if experiment.startswith(
            "E5 - RAC v2 fine-tuned"
        ):


            encoder_params = list(

                model
                .gae_encoder
                .lin_encode
                .parameters()
            )


            encoder_param_ids = {

                id(p)
                for p
                in encoder_params
            }


            other_params = [

                p

                for p
                in model.parameters()

                if (
                    p.requires_grad
                    and
                    id(p)
                    not in encoder_param_ids
                )
            ]


            optimizer = (
                torch.optim.Adam(

                    [

                        {
                            "params":
                                encoder_params,

                            "lr":
                                1e-5
                        },

                        {
                            "params":
                                other_params,

                            "lr":
                                LEARNING_RATE
                        }
                    ],

                    weight_decay=
                        WEIGHT_DECAY
                )
            )


        else:

            optimizer = (
                torch.optim.Adam(

                    trainable_parameters,

                    lr=
                        LEARNING_RATE,

                    weight_decay=
                        WEIGHT_DECAY
                )
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
            nn.BCELoss()
        )


        # ----------------------------------------------------
        # early stopping
        # ----------------------------------------------------

        best_val_loss = (
            float("inf")
        )


        best_epoch = 0

        best_state = None

        patience_counter = 0


        # ====================================================
        # TRAINING
        # ====================================================

        for epoch in range(

            1,

            MAX_EPOCHS + 1
        ):


            set_training_mode(

                experiment,

                model
            )


            total_train_loss = (
                0.0
            )


            for data in (
                train_loader
            ):


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


            train_loss = (

                total_train_loss
                /
                len(
                    train_loader
                )
            )


            # ------------------------------------------------
            # validation
            # ------------------------------------------------

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


            # ------------------------------------------------
            # best checkpoint according to validation loss
            # ------------------------------------------------

            if (
                val_loss
                <
                best_val_loss
            ):


                best_val_loss = (
                    val_loss
                )


                best_epoch = (
                    epoch
                )


                patience_counter = 0


                best_state = {

                    key:
                        value
                        .detach()
                        .cpu()
                        .clone()

                    for key, value
                    in (
                        model
                        .state_dict()
                        .items()
                    )
                }


            else:

                patience_counter += 1


            # ------------------------------------------------
            # progress
            # ------------------------------------------------

            if (
                epoch % 10
                == 0
            ):


                metrics_now = (
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
                    f"{train_loss:.4f}"

                    f" | Val Loss: "
                    f"{val_loss:.4f}"

                    f" | Val Acc: "
                    f"{metrics_now['accuracy_percent']:.2f}%"
                )


            # ------------------------------------------------
            # early stop
            # ------------------------------------------------

            if (
                patience_counter
                >=
                EARLY_STOPPING_PATIENCE
            ):


                print(

                    f"Early stopping "
                    f"at epoch "
                    f"{epoch}"
                )


                break


        # ====================================================
        # RESTORE BEST MODEL
        # ====================================================

        if (
            best_state
            is not None
        ):

            model.load_state_dict(
                best_state
            )


        model.to(
            device
        )

        model.eval()


        # ====================================================
        # FINAL VALIDATION
        # ====================================================

        final_labels = []

        final_probs = []

        final_subjects = []


        with torch.no_grad():


            for row_idx, data in enumerate(
                val_loader
            ):


                data = data.to(
                    device
                )


                pred = (
                    forward_model(

                        experiment,

                        model,

                        data
                    )
                )


                probability = float(

                    pred
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
                        row_idx
                    ][
                        "subject_id"
                    ]
                )


        metrics = (
            calculate_metrics(

                final_labels,

                final_probs
            )
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


        # ====================================================
        # SAVE FOLD RESULT
        # ====================================================

        all_results.append({

            "experiment":
                experiment,

            "K":
                get_k(
                    experiment
                ),

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

            "accuracy_percent":
                metrics[
                    "accuracy_percent"
                ],

            "roc_auc":
                metrics[
                    "roc_auc"
                ],

            "precision":
                metrics[
                    "precision"
                ],

            "sensitivity":
                metrics[
                    "sensitivity"
                ],

            "specificity":
                metrics[
                    "specificity"
                ],

            "f1":
                metrics[
                    "f1"
                ],

            "tn":
                metrics["tn"],

            "fp":
                metrics["fp"],

            "fn":
                metrics["fn"],

            "tp":
                metrics["tp"]
        })


        # ----------------------------------------------------
        # subject predictions
        # ----------------------------------------------------

        for (
            subject,
            label,
            probability,
            predicted_class

        ) in zip(

            final_subjects,

            final_labels,

            final_probs,

            metrics["classes"]
        ):


            all_predictions.append({

                "experiment":
                    experiment,

                "K":
                    get_k(
                        experiment
                    ),

                "repeat":
                    repeat_number,

                "fold":
                    fold_number,

                "subject_id":
                    subject,

                "ground_truth":
                    label,

                "probability":
                    probability,

                "predicted_class":
                    predicted_class
            })


# ============================================================
# 27. DATAFRAMES
# ============================================================

results_df = (
    pd.DataFrame(
        all_results
    )
)


predictions_df = (
    pd.DataFrame(
        all_predictions
    )
)


# ============================================================
# 28. SUMMARY
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
# 29. SAVE RESULTS
# ============================================================

raw_file = (
    RESULTS_DIR /
    "sana06_all_folds.csv"
)


summary_file = (
    RESULTS_DIR /
    "sana06_summary.csv"
)


prediction_file = (
    RESULTS_DIR /
    "sana06_predictions.csv"
)


results_df.to_csv(

    raw_file,

    index=False
)


summary_df.to_csv(

    summary_file,

    index=False
)


predictions_df.to_csv(

    prediction_file,

    index=False
)


# ============================================================
# 30. PRINT FINAL SUMMARY
# ============================================================

print("\n\n")

print("=" * 100)

print(
    "SANA 06 - IMPROVED ABLATION SUMMARY"
)

print("=" * 100)


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
    ]

    .to_string(
        index=False
    )
)


print("\n")

print(
    "Results saved in:"
)

print(
    RESULTS_DIR
)