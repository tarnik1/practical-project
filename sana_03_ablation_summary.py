# ============================================================
# sana_03_ablation_summary.py
#
# Combines:
#
# E0 = Ab-initio baseline
# E1 = Pretrained encoder only
# E2 = Pretrained + FAISS retrieval mean
# E3 = RAC v1 dot-product attention
# E4 = RAC v2 learnable linear attention
#
# This script DOES NOT train any model.
#
# It generates:
#   - final ablation summary CSV
#   - incremental improvement CSV
# ============================================================

import os
import numpy as np
import pandas as pd

from sklearn.metrics import (
    accuracy_score,
    roc_auc_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
)


# ------------------------------------------------------------
# 1. Paths
# ------------------------------------------------------------

output_dir = r'C:\Users\imam\Desktop\Tara Project\processed_data_1'

results_dir = os.path.join(
    output_dir,
    "sana_results"
)

os.makedirs(
    results_dir,
    exist_ok=True
)


pretrained_summary_path = os.path.join(
    results_dir,
    "sana_pretrained_only_summary.csv"
)

retrieval_summary_path = os.path.join(
    results_dir,
    "sana_retrieval_mean_summary.csv"
)

original_eval_path = os.path.join(
    output_dir,
    "test_evaluation_results.csv"
)


# ------------------------------------------------------------
# 2. Check required files
# ------------------------------------------------------------

required_files = [
    pretrained_summary_path,
    retrieval_summary_path,
    original_eval_path,
]

for path in required_files:

    if not os.path.exists(path):

        raise FileNotFoundError(
            f"\nRequired file not found:\n{path}\n"
        )


# ------------------------------------------------------------
# 3. Load Sana's two experiment summaries
# ------------------------------------------------------------

pretrained_df = pd.read_csv(
    pretrained_summary_path
)

retrieval_df = pd.read_csv(
    retrieval_summary_path
)


# ------------------------------------------------------------
# 4. Load original repository evaluation
# ------------------------------------------------------------

original_df = pd.read_csv(
    original_eval_path
)


# ------------------------------------------------------------
# Helper function for metric calculation
# ------------------------------------------------------------

def calculate_metrics(
    labels,
    probabilities,
    predictions,
):

    labels = np.array(labels)
    probabilities = np.array(probabilities)
    predictions = np.array(predictions)

    accuracy = accuracy_score(
        labels,
        predictions
    )

    precision = precision_score(
        labels,
        predictions,
        zero_division=0
    )

    recall = recall_score(
        labels,
        predictions,
        zero_division=0
    )

    f1 = f1_score(
        labels,
        predictions,
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
        predictions
    )

    tn, fp, fn, tp = cm.ravel()

    specificity = (
        tn / (tn + fp)
        if (tn + fp) > 0
        else np.nan
    )

    return {
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
    }


# ------------------------------------------------------------
# 5. Calculate original model metrics
# ------------------------------------------------------------

labels = original_df[
    "ground_truth"
].values


# ----- Ab-initio baseline -----

baseline_metrics = calculate_metrics(

    labels,

    original_df[
        "baseline_prob"
    ].values,

    original_df[
        "baseline_predicted_class"
    ].values,
)


# ----- RAC v1 -----

rac_v1_metrics = calculate_metrics(

    labels,

    original_df[
        "rac_v1_prob"
    ].values,

    original_df[
        "rac_v1_predicted"
    ].values,
)


# ----- RAC v2 -----

rac_v2_metrics = calculate_metrics(

    labels,

    original_df[
        "rac_v2_prob"
    ].values,

    original_df[
        "rac_v2_predicted"
    ].values,
)


# ------------------------------------------------------------
# 6. Extract Sana experiment metrics
# ------------------------------------------------------------

pretrained_row = (
    pretrained_df.iloc[0]
)

retrieval_row = (
    retrieval_df.iloc[0]
)


# ------------------------------------------------------------
# Helper for building summary rows
# ------------------------------------------------------------

def make_row(
    experiment,
    pretraining,
    retrieval,
    attention,
    metrics,
):

    return {

        "experiment":
            experiment,

        "external_pretraining":
            pretraining,

        "retrieval":
            retrieval,

        "attention":
            attention,

        "accuracy":
            metrics["accuracy"],

        "accuracy_percent":
            metrics["accuracy"] * 100,

        "roc_auc":
            metrics["roc_auc"],

        "precision":
            metrics["precision"],

        "recall_sensitivity":
            metrics["recall_sensitivity"],

        "specificity":
            metrics["specificity"],

        "f1_score":
            metrics["f1_score"],

        "TN":
            metrics["TN"],

        "FP":
            metrics["FP"],

        "FN":
            metrics["FN"],

        "TP":
            metrics["TP"],
    }


# ------------------------------------------------------------
# Convert Sana CSV rows into dictionaries with same structure
# ------------------------------------------------------------

pretrained_metrics = {

    "accuracy":
        pretrained_row["accuracy"],

    "roc_auc":
        pretrained_row["roc_auc"],

    "precision":
        pretrained_row["precision"],

    "recall_sensitivity":
        pretrained_row["recall_sensitivity"],

    "specificity":
        pretrained_row["specificity"],

    "f1_score":
        pretrained_row["f1_score"],

    "TN":
        pretrained_row["TN"],

    "FP":
        pretrained_row["FP"],

    "FN":
        pretrained_row["FN"],

    "TP":
        pretrained_row["TP"],
}


retrieval_metrics = {

    "accuracy":
        retrieval_row["accuracy"],

    "roc_auc":
        retrieval_row["roc_auc"],

    "precision":
        retrieval_row["precision"],

    "recall_sensitivity":
        retrieval_row["recall_sensitivity"],

    "specificity":
        retrieval_row["specificity"],

    "f1_score":
        retrieval_row["f1_score"],

    "TN":
        retrieval_row["TN"],

    "FP":
        retrieval_row["FP"],

    "FN":
        retrieval_row["FN"],

    "TP":
        retrieval_row["TP"],
}


# ------------------------------------------------------------
# 7. Final ablation table
# ------------------------------------------------------------

rows = []


rows.append(
    make_row(
        experiment="E0 - Ab-initio",
        pretraining="No",
        retrieval="No",
        attention="No",
        metrics=baseline_metrics,
    )
)


rows.append(
    make_row(
        experiment="E1 - Pretrained only",
        pretraining="Yes",
        retrieval="No",
        attention="No",
        metrics=pretrained_metrics,
    )
)


rows.append(
    make_row(
        experiment="E2 - Retrieval mean",
        pretraining="Yes",
        retrieval="Yes",
        attention="No / equal averaging",
        metrics=retrieval_metrics,
    )
)


rows.append(
    make_row(
        experiment="E3 - RAC v1",
        pretraining="Yes",
        retrieval="Yes",
        attention="Dot-product",
        metrics=rac_v1_metrics,
    )
)


rows.append(
    make_row(
        experiment="E4 - RAC v2",
        pretraining="Yes",
        retrieval="Yes",
        attention="Learnable linear",
        metrics=rac_v2_metrics,
    )
)


summary_df = pd.DataFrame(
    rows
)


# ------------------------------------------------------------
# 8. Calculate incremental changes
#
# IMPORTANT:
# These are descriptive percentage-point differences.
# They should not automatically be called causal
# "contributions" unless the experimental design supports it.
# ------------------------------------------------------------

summary_df[
    "change_from_previous_pp"
] = (
    summary_df[
        "accuracy_percent"
    ].diff()
)


summary_df[
    "change_from_ab_initio_pp"
] = (
    summary_df[
        "accuracy_percent"
    ]
    -
    summary_df.loc[
        0,
        "accuracy_percent"
    ]
)


# ------------------------------------------------------------
# 9. Save full summary
# ------------------------------------------------------------

summary_path = os.path.join(
    results_dir,
    "sana_ablation_summary.csv"
)

summary_df.to_csv(
    summary_path,
    index=False
)


# ------------------------------------------------------------
# 10. Save a compact poster table
# ------------------------------------------------------------

poster_df = summary_df[[
    "experiment",
    "external_pretraining",
    "retrieval",
    "attention",
    "accuracy_percent",
    "roc_auc",
    "change_from_previous_pp",
]].copy()


poster_df["accuracy_percent"] = (
    poster_df[
        "accuracy_percent"
    ].round(2)
)

poster_df["roc_auc"] = (
    poster_df[
        "roc_auc"
    ].round(3)
)

poster_df[
    "change_from_previous_pp"
] = (
    poster_df[
        "change_from_previous_pp"
    ].round(2)
)


poster_path = os.path.join(
    results_dir,
    "sana_ablation_poster_table.csv"
)

poster_df.to_csv(
    poster_path,
    index=False
)


# ------------------------------------------------------------
# 11. Save incremental comparison separately
# ------------------------------------------------------------

incremental_rows = []

for i in range(
    1,
    len(summary_df)
):

    incremental_rows.append({

        "from_experiment":
            summary_df.loc[
                i - 1,
                "experiment"
            ],

        "to_experiment":
            summary_df.loc[
                i,
                "experiment"
            ],

        "from_accuracy_percent":
            summary_df.loc[
                i - 1,
                "accuracy_percent"
            ],

        "to_accuracy_percent":
            summary_df.loc[
                i,
                "accuracy_percent"
            ],

        "difference_percentage_points":
            summary_df.loc[
                i,
                "accuracy_percent"
            ]
            -
            summary_df.loc[
                i - 1,
                "accuracy_percent"
            ],
    })


incremental_df = pd.DataFrame(
    incremental_rows
)

incremental_df.to_csv(
    os.path.join(
        results_dir,
        "sana_incremental_improvements.csv"
    ),
    index=False
)


# ------------------------------------------------------------
# 12. Display results
# ------------------------------------------------------------

print(
    "\n=============================================="
)

print(
    "FINAL RAC ABLATION SUMMARY"
)

print(
    "==============================================\n"
)


display_columns = [
    "experiment",
    "accuracy_percent",
    "roc_auc",
    "change_from_previous_pp",
]


print(
    summary_df[
        display_columns
    ].to_string(
        index=False
    )
)


print(
    "\n----------------------------------------------"
)

print(
    "Incremental changes in accuracy"
)

print(
    "----------------------------------------------\n"
)


print(
    incremental_df.to_string(
        index=False
    )
)


print(
    "\nFiles saved:"
)

print(summary_path)
print(poster_path)

print(
    os.path.join(
        results_dir,
        "sana_incremental_improvements.csv"
    )
)