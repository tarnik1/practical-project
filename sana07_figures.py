# ============================================================
# SANA 07
# PUBLICATION-QUALITY SCIENTIFIC FIGURES
# ============================================================
#
# Input:
#   sana06_all_folds.csv
#
# Output:
#   High-resolution PNG, PDF and SVG figures
#
# Figures:
#   1. Accuracy distribution across all experiments
#   2. Mean accuracy + 95% confidence intervals
#   3. ROC-AUC distribution
#   4. Sensitivity vs specificity
#   5. Effect of retrieval K
#   6. Combined 4-panel publication figure
#
# ============================================================

import os
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib

# Important for HPC / no graphical display
matplotlib.use("Agg")

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D


# ============================================================
# 1. PATHS
# ============================================================

# Possible locations of sana_results on your HPC
possible_results_dirs = [

    Path(
        "C:/Users/nuji0350/Desktop/Tara Project/processed_data_1/sana_results"
    ),

    Path(
        "C:/Users/nuji0350/Desktop/Tara Project/processed_data_1/sana_results"
    ),
]


RESULTS_DIR = None

for candidate in possible_results_dirs:

    if (
        candidate.exists()
        and
        (candidate / "sana06_all_folds.csv").exists()
    ):
        RESULTS_DIR = candidate
        break


if RESULTS_DIR is None:

    raise FileNotFoundError(
        "Could not find sana06_all_folds.csv.\n"
        "Checked:\n"
        + "\n".join(str(x) for x in possible_results_dirs)
    )


INPUT_FILE = (
    RESULTS_DIR /
    "sana06_all_folds.csv"
)

FIGURE_DIR = (
    RESULTS_DIR /
    "scientific_figures"
)

FIGURE_DIR.mkdir(
    parents=True,
    exist_ok=True
)


print("Reading:")
print(INPUT_FILE)

print("\nFigures will be saved in:")
print(FIGURE_DIR)


# ============================================================
# 2. LOAD DATA
# ============================================================

df = pd.read_csv(
    INPUT_FILE
)


required_columns = [
    "experiment",
    "accuracy_percent",
    "roc_auc",
    "sensitivity",
    "specificity",
    "f1"
]


for column in required_columns:

    if column not in df.columns:

        raise ValueError(
            f"Missing required column: {column}"
        )


# ============================================================
# 3. SHORT DISPLAY NAMES
# ============================================================

label_map = {

    "E0 - Ab-initio":
        "Ab-initio",

    "E1 - Pretrained only":
        "Pretrained\nonly",

    "E2 - Retrieval mean (K=5)":
        "Retrieval mean\nK=5",

    "E2 - Retrieval mean (K=10)":
        "Retrieval mean\nK=10",

    "E2 - Retrieval mean (K=15)":
        "Retrieval mean\nK=15",

    "E3 - RAC v1 (K=5)":
        "RAC v1\nK=5",

    "E3 - RAC v1 (K=10)":
        "RAC v1\nK=10",

    "E3 - RAC v1 (K=15)":
        "RAC v1\nK=15",

    "E4 - RAC v2 frozen (K=5)":
        "RAC v2 frozen\nK=5",

    "E4 - RAC v2 frozen (K=10)":
        "RAC v2 frozen\nK=10",

    "E4 - RAC v2 frozen (K=15)":
        "RAC v2 frozen\nK=15",

    "E5 - RAC v2 fine-tuned (K=5)":
        "RAC v2 fine-tuned\nK=5",

    "E5 - RAC v2 fine-tuned (K=10)":
        "RAC v2 fine-tuned\nK=10",

    "E5 - RAC v2 fine-tuned (K=15)":
        "RAC v2 fine-tuned\nK=15",
}


experiment_order = [

    "E0 - Ab-initio",
    "E1 - Pretrained only",

    "E2 - Retrieval mean (K=5)",
    "E2 - Retrieval mean (K=10)",
    "E2 - Retrieval mean (K=15)",

    "E3 - RAC v1 (K=5)",
    "E3 - RAC v1 (K=10)",
    "E3 - RAC v1 (K=15)",

    "E4 - RAC v2 frozen (K=5)",
    "E4 - RAC v2 frozen (K=10)",
    "E4 - RAC v2 frozen (K=15)",

    "E5 - RAC v2 fine-tuned (K=5)",
    "E5 - RAC v2 fine-tuned (K=10)",
    "E5 - RAC v2 fine-tuned (K=15)",
]


# Keep only experiments actually available
experiment_order = [
    x for x in experiment_order
    if x in df["experiment"].unique()
]


# ============================================================
# 4. GENERAL FIGURE SETTINGS
# ============================================================

plt.rcParams.update({

    "font.size": 10,

    "axes.titlesize": 13,

    "axes.labelsize": 11,

    "xtick.labelsize": 9,

    "ytick.labelsize": 9,

    "legend.fontsize": 9,

    "figure.dpi": 120,

    "savefig.dpi": 600,

    "axes.spines.top": False,

    "axes.spines.right": False,

    "axes.linewidth": 0.8,

    "pdf.fonttype": 42,

    "ps.fonttype": 42,
})


# ============================================================
# 5. HELPER: SAVE FIGURE
# ============================================================

def save_figure(
    fig,
    filename
):

    png = FIGURE_DIR / f"{filename}.png"
    pdf = FIGURE_DIR / f"{filename}.pdf"
    svg = FIGURE_DIR / f"{filename}.svg"

    fig.savefig(
        png,
        bbox_inches="tight"
    )

    fig.savefig(
        pdf,
        bbox_inches="tight"
    )

    fig.savefig(
        svg,
        bbox_inches="tight"
    )

    print(
        f"Saved: {filename}"
    )


# ============================================================
# 6. HELPER: DISTRIBUTION PLOT
# ============================================================

def distribution_plot(
    ax,
    dataframe,
    metric,
    ylabel,
    title,
    ylim=None
):

    groups = []

    labels = []

    for experiment in experiment_order:

        values = (

            dataframe.loc[
                dataframe["experiment"] == experiment,
                metric
            ]
            .dropna()
            .values
        )

        groups.append(
            values
        )

        labels.append(
            label_map.get(
                experiment,
                experiment
            )
        )


    positions = np.arange(
        1,
        len(groups) + 1
    )


    # --------------------------------------------------------
    # Violin plots
    # --------------------------------------------------------

    violin = ax.violinplot(

        groups,

        positions=positions,

        widths=0.75,

        showmeans=False,

        showmedians=False,

        showextrema=False
    )


    for body in violin["bodies"]:

        body.set_alpha(
            0.25
        )

        body.set_edgecolor(
            "black"
        )

        body.set_linewidth(
            0.8
        )


    # --------------------------------------------------------
    # Boxplots
    # --------------------------------------------------------

    box = ax.boxplot(

        groups,

        positions=positions,

        widths=0.22,

        patch_artist=True,

        showfliers=False,

        medianprops={
            "linewidth": 1.5
        },

        boxprops={
            "linewidth": 1.0
        },

        whiskerprops={
            "linewidth": 1.0
        },

        capprops={
            "linewidth": 1.0
        }
    )


    for patch in box["boxes"]:

        patch.set_alpha(
            0.65
        )


    # --------------------------------------------------------
    # Individual CV fold points
    # --------------------------------------------------------

    rng = np.random.default_rng(
        42
    )


    for position, values in zip(
        positions,
        groups
    ):

        jitter = rng.normal(

            loc=0,

            scale=0.055,

            size=len(values)
        )


        ax.scatter(

            position + jitter,

            values,

            s=14,

            alpha=0.40,

            zorder=3
        )


        # Mean marker
        mean_value = np.mean(
            values
        )


        ax.scatter(

            position,

            mean_value,

            marker="D",

            s=42,

            edgecolor="black",

            linewidth=0.8,

            zorder=5
        )


    ax.set_xticks(
        positions
    )

    ax.set_xticklabels(
        labels,
        rotation=55,
        ha="right"
    )

    ax.set_ylabel(
        ylabel
    )

    ax.set_title(
        title,
        loc="left",
        fontweight="bold"
    )


    if ylim is not None:

        ax.set_ylim(
            ylim
        )


    ax.grid(
        axis="y",
        alpha=0.2
    )


# ============================================================
# FIGURE 1
# ACCURACY DISTRIBUTIONS
# ============================================================

fig, ax = plt.subplots(
    figsize=(16, 7)
)


distribution_plot(

    ax=ax,

    dataframe=df,

    metric="accuracy_percent",

    ylabel="Validation accuracy (%)",

    title=(
        "Distribution of validation accuracy "
        "across repeated cross-validation folds"
    ),

    ylim=(0, 105)
)


ax.axhline(
    50,
    linestyle="--",
    linewidth=1,
    alpha=0.6
)


ax.text(

    0.01,
    0.51,

    "50%",

    transform=ax.transAxes,

    fontsize=9
)


legend_items = [

    Line2D(
        [0],
        [0],
        marker="o",
        linestyle="None",
        markersize=6,
        label="Individual validation fold"
    ),

    Line2D(
        [0],
        [0],
        marker="D",
        linestyle="None",
        markersize=7,
        markeredgecolor="black",
        label="Mean"
    )
]


ax.legend(
    handles=legend_items,
    frameon=False,
    loc="upper right"
)


fig.tight_layout()


save_figure(
    fig,
    "Figure1_accuracy_distributions"
)


plt.close(
    fig
)


# ============================================================
# 7. SUMMARY STATISTICS WITH 95% CI
# ============================================================

summary_rows = []


for experiment in experiment_order:

    subset = df[
        df["experiment"] == experiment
    ]


    values = (
        subset[
            "accuracy_percent"
        ]
        .dropna()
        .values
    )


    n = len(values)

    mean = np.mean(
        values
    )

    sd = np.std(
        values,
        ddof=1
    )


    # Approximate 95% CI
    se = sd / np.sqrt(
        n
    )

    ci95 = 1.96 * se


    summary_rows.append({

        "experiment":
            experiment,

        "label":
            label_map.get(
                experiment,
                experiment
            ).replace("\n", " "),

        "mean":
            mean,

        "sd":
            sd,

        "n":
            n,

        "ci95":
            ci95
    })


summary = pd.DataFrame(
    summary_rows
)


# ============================================================
# FIGURE 2
# MEAN ACCURACY + 95% CI
# ============================================================

fig, ax = plt.subplots(
    figsize=(10, 8)
)


y_positions = np.arange(
    len(summary)
)


ax.errorbar(

    summary["mean"],

    y_positions,

    xerr=summary["ci95"],

    fmt="o",

    markersize=7,

    capsize=4,

    linewidth=1.5
)


ax.set_yticks(
    y_positions
)


ax.set_yticklabels(
    summary["label"]
)


ax.invert_yaxis()


ax.axvline(

    50,

    linestyle="--",

    linewidth=1,

    alpha=0.6
)


ax.set_xlabel(
    "Mean validation accuracy (%)"
)


ax.set_title(

    "Model comparison with 95% confidence intervals",

    loc="left",

    fontweight="bold"
)


ax.grid(
    axis="x",
    alpha=0.2
)


# Add actual mean numbers
for y, mean in zip(
    y_positions,
    summary["mean"]
):

    ax.text(

        mean + 1,

        y,

        f"{mean:.1f}%",

        va="center",

        fontsize=8
    )


fig.tight_layout()


save_figure(
    fig,
    "Figure2_mean_accuracy_95CI"
)


plt.close(
    fig
)


# ============================================================
# FIGURE 3
# ROC-AUC DISTRIBUTIONS
# ============================================================

fig, ax = plt.subplots(
    figsize=(16, 7)
)


distribution_plot(

    ax=ax,

    dataframe=df,

    metric="roc_auc",

    ylabel="ROC-AUC",

    title=(
        "Distribution of ROC-AUC "
        "across repeated cross-validation folds"
    ),

    ylim=(0, 1.05)
)


ax.axhline(

    0.5,

    linestyle="--",

    linewidth=1,

    alpha=0.6
)


fig.tight_layout()


save_figure(
    fig,
    "Figure3_auc_distributions"
)


plt.close(
    fig
)


# ============================================================
# FIGURE 4
# SENSITIVITY VS SPECIFICITY
# ============================================================

performance = (

    df

    .groupby(
        "experiment",
        sort=False
    )

    .agg(

        sensitivity=(
            "sensitivity",
            "mean"
        ),

        specificity=(
            "specificity",
            "mean"
        ),

        accuracy=(
            "accuracy_percent",
            "mean"
        )
    )

    .reset_index()
)


fig, ax = plt.subplots(
    figsize=(9, 8)
)


ax.scatter(

    performance[
        "specificity"
    ],

    performance[
        "sensitivity"
    ],

    s=75
)


for _, row in performance.iterrows():

    short_label = (
        label_map.get(
            row["experiment"],
            row["experiment"]
        )
        .replace("\n", " ")
    )


    ax.annotate(

        short_label,

        (
            row["specificity"],
            row["sensitivity"]
        ),

        xytext=(5, 5),

        textcoords="offset points",

        fontsize=7
    )


ax.plot(

    [0, 1],

    [0, 1],

    linestyle="--",

    linewidth=1,

    alpha=0.4
)


ax.set_xlim(
    0.25,
    1.0
)


ax.set_ylim(
    0.20,
    1.0
)


ax.set_xlabel(
    "Mean specificity"
)


ax.set_ylabel(
    "Mean sensitivity"
)


ax.set_title(

    "Sensitivity–specificity profile of each model",

    loc="left",

    fontweight="bold"
)


ax.grid(
    alpha=0.2
)


fig.tight_layout()


save_figure(
    fig,
    "Figure4_sensitivity_specificity"
)


plt.close(
    fig
)


# ============================================================
# 8. PREPARE K ANALYSIS
# ============================================================

k_df = df[
    df["experiment"].str.contains(
        "K=",
        regex=False
    )
].copy()


def extract_k(
    name
):

    return int(
        name
        .split("K=")[-1]
        .split(")")[0]
    )


def model_family(
    name
):

    if name.startswith(
        "E2"
    ):

        return "Retrieval mean"

    if name.startswith(
        "E3"
    ):

        return "RAC v1"

    if name.startswith(
        "E4"
    ):

        return "RAC v2 frozen"

    if name.startswith(
        "E5"
    ):

        return "RAC v2 fine-tuned"

    return name


k_df["K"] = (
    k_df["experiment"]
    .apply(
        extract_k
    )
)


k_df["model_family"] = (
    k_df["experiment"]
    .apply(
        model_family
    )
)


k_summary = (

    k_df

    .groupby(
        [
            "model_family",
            "K"
        ]
    )

    .agg(

        mean_accuracy=(
            "accuracy_percent",
            "mean"
        ),

        sd_accuracy=(
            "accuracy_percent",
            "std"
        ),

        n=(
            "accuracy_percent",
            "count"
        )
    )

    .reset_index()
)


k_summary[
    "ci95"
] = (

    1.96
    *
    k_summary[
        "sd_accuracy"
    ]
    /
    np.sqrt(
        k_summary[
            "n"
        ]
    )
)


# ============================================================
# FIGURE 5
# EFFECT OF K
# ============================================================

fig, ax = plt.subplots(
    figsize=(9, 6)
)


family_order = [

    "Retrieval mean",

    "RAC v1",

    "RAC v2 frozen",

    "RAC v2 fine-tuned"
]


for family in family_order:

    subset = (

        k_summary[
            k_summary[
                "model_family"
            ] == family
        ]
        .sort_values(
            "K"
        )
    )


    ax.errorbar(

        subset["K"],

        subset[
            "mean_accuracy"
        ],

        yerr=subset[
            "ci95"
        ],

        marker="o",

        markersize=7,

        capsize=4,

        linewidth=1.8,

        label=family
    )


ax.set_xticks(
    [5, 10, 15]
)


ax.set_xlabel(
    "Number of retrieved neighbours (K)"
)


ax.set_ylabel(
    "Mean validation accuracy (%)"
)


ax.set_title(

    "Effect of retrieval-set size on classification performance",

    loc="left",

    fontweight="bold"
)


ax.legend(
    frameon=False
)


ax.grid(
    alpha=0.2
)


fig.tight_layout()


save_figure(
    fig,
    "Figure5_K_comparison"
)


plt.close(
    fig
)


# ============================================================
# FIGURE 6
# COMBINED PUBLICATION FIGURE
# ============================================================

fig = plt.figure(
    figsize=(16, 13)
)


grid = fig.add_gridspec(
    2,
    2,
    hspace=0.40,
    wspace=0.30
)


# ------------------------------------------------------------
# Panel A
# Accuracy distribution
# ------------------------------------------------------------

axA = fig.add_subplot(
    grid[0, 0]
)


distribution_plot(

    axA,

    df,

    "accuracy_percent",

    "Validation accuracy (%)",

    "A  Accuracy distributions",

    ylim=(0, 105)
)


# Smaller labels for combined figure
axA.tick_params(
    axis="x",
    labelsize=6.5
)


# ------------------------------------------------------------
# Panel B
# AUC distribution
# ------------------------------------------------------------

axB = fig.add_subplot(
    grid[0, 1]
)


distribution_plot(

    axB,

    df,

    "roc_auc",

    "ROC-AUC",

    "B  ROC-AUC distributions",

    ylim=(0, 1.05)
)


axB.tick_params(
    axis="x",
    labelsize=6.5
)


# ------------------------------------------------------------
# Panel C
# K comparison
# ------------------------------------------------------------

axC = fig.add_subplot(
    grid[1, 0]
)


for family in family_order:

    subset = (

        k_summary[
            k_summary[
                "model_family"
            ] == family
        ]
        .sort_values(
            "K"
        )
    )


    axC.errorbar(

        subset["K"],

        subset[
            "mean_accuracy"
        ],

        yerr=subset[
            "ci95"
        ],

        marker="o",

        markersize=6,

        capsize=3,

        linewidth=1.5,

        label=family
    )


axC.set_xticks(
    [5, 10, 15]
)


axC.set_xlabel(
    "Number of retrieved neighbours (K)"
)


axC.set_ylabel(
    "Mean validation accuracy (%)"
)


axC.set_title(

    "C  Effect of retrieval-set size",

    loc="left",

    fontweight="bold"
)


axC.legend(
    frameon=False,
    fontsize=8
)


axC.grid(
    alpha=0.2
)


# ------------------------------------------------------------
# Panel D
# Sensitivity / specificity
# ------------------------------------------------------------

axD = fig.add_subplot(
    grid[1, 1]
)


x = np.arange(
    len(performance)
)


width = 0.36


axD.bar(

    x - width / 2,

    performance[
        "sensitivity"
    ],

    width,

    label="Sensitivity"
)


axD.bar(

    x + width / 2,

    performance[
        "specificity"
    ],

    width,

    label="Specificity"
)


axD.set_xticks(
    x
)


axD.set_xticklabels(

    [
        label_map.get(
            x,
            x
        )
        for x in performance[
            "experiment"
        ]
    ],

    rotation=55,

    ha="right",

    fontsize=6.5
)


axD.set_ylim(
    0,
    1
)


axD.set_ylabel(
    "Mean score"
)


axD.set_title(

    "D  Class-specific performance",

    loc="left",

    fontweight="bold"
)


axD.legend(
    frameon=False
)


axD.grid(
    axis="y",
    alpha=0.2
)


fig.suptitle(

    "Ablation analysis of retrieval-augmented classification",

    fontsize=17,

    fontweight="bold",

    y=0.995
)


save_figure(
    fig,
    "Figure6_combined_publication_figure"
)


plt.close(
    fig
)


# ============================================================
# 9. SAVE SUMMARY TABLE USED FOR PLOTS
# ============================================================

summary.to_csv(

    FIGURE_DIR /
    "figure_accuracy_summary.csv",

    index=False
)


k_summary.to_csv(

    FIGURE_DIR /
    "figure_K_summary.csv",

    index=False
)


# ============================================================
# DONE
# ============================================================

print("\n" + "=" * 70)

print(
    "ALL FIGURES CREATED SUCCESSFULLY"
)

print("=" * 70)

print(
    "\nOutput folder:"
)

print(
    FIGURE_DIR
)

print(
    "\nRecommended main figure for paper/poster:"
)

print(
    "Figure6_combined_publication_figure.pdf"
)