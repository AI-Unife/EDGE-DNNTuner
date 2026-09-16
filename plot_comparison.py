"""
Compare model distributions (Accuracy, Params, FLOPs) across datasets
for two conditions:
    1) "All models"                  -> summary_best_results_tot_w_flops.csv
    2) "ReLU/SELU < 300k params"      -> gesture_vs_roigesture_relu_300k_merged.xlsx (sheet "Clean_Data")

Produces a 1x3 grid of boxplots (Accuracy / Params / FLOPs), grouped by
dataset, with the two conditions shown side by side.
"""

import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

# ---------------------------------------------------------------------------
# 1. File paths (edit if needed)
# ---------------------------------------------------------------------------
CSV_PATH = "summary_best_results_tot_w_flops.csv"
XLSX_PATH = "../gesture_vs_roigesture_relu_300k_merged.xlsx"
XLSX_SHEET = "Clean_Data"

OUTPUT_PATH = "confronto_distribuzioni.png"

# ---------------------------------------------------------------------------
# 2. Load and normalize both sources into a common schema:
#    columns -> dataset, accuracy (0-100), params, flops, case
# ---------------------------------------------------------------------------
csv_df = pd.read_csv(CSV_PATH)
csv_df = csv_df[["dataset", "best_accuracy", "best_nparams", "best_flops"]].copy()
csv_df.columns = ["dataset", "accuracy", "params", "flops"]
csv_df["accuracy"] = csv_df["accuracy"] * 100          # scale 0-1 -> 0-100
csv_df["case"] = "All models"

xlsx_df = pd.read_excel(XLSX_PATH, sheet_name=XLSX_SHEET)
xlsx_df = xlsx_df[["Dataset", "Accuracy (%)", "Params", "FLOPs"]].copy()
xlsx_df.columns = ["dataset", "accuracy", "params", "flops"]
xlsx_df["case"] = "ReLU/SELU < 300k params"

df = pd.concat([csv_df, xlsx_df], ignore_index=True)

# ---------------------------------------------------------------------------
# 3. Plot settings
# ---------------------------------------------------------------------------
datasets = ["gesture", "roigesture_coords", "roigesture_matrix"]
cases = ["All models", "ReLU/SELU < 300k params"]
colors = {
    "All models": "#4C72B0",
    "ReLU/SELU < 300k params": "#DD8452",
}
metrics = [
    ("accuracy", "Accuracy (%)", False),   # (column, axis title, use log scale)
    ("params", "Params (n)", True),
    ("flops", "FLOPs", True),
]

fig, axes = plt.subplots(1, 3, figsize=(16, 6))

for ax, (col, label, logscale) in zip(axes, metrics):
    positions, data_to_plot, box_colors = [], [], []
    xticks, xticklabels = [], []
    pos, width = 0, 0.35

    for ds in datasets:
        for j, case in enumerate(cases):
            values = df[(df["dataset"] == ds) & (df["case"] == case)][col].dropna().values
            offset = (j - 0.5) * width
            positions.append(pos + offset)
            data_to_plot.append(values)
            box_colors.append(colors[case])
        xticks.append(pos)
        xticklabels.append(ds)
        pos += 1.2

    bp = ax.boxplot(
        data_to_plot,
        positions=positions,
        widths=width * 0.9,
        patch_artist=True,
        showfliers=True,
        medianprops=dict(color="black"),
    )
    for patch, c in zip(bp["boxes"], box_colors):
        patch.set_facecolor(c)
        patch.set_alpha(0.75)

    ax.set_xticks(xticks)
    ax.set_xticklabels(xticklabels, rotation=15, ha="right")
    ax.set_title(label, fontsize=13, fontweight="bold")
    if logscale:
        ax.set_yscale("log")
    ax.grid(axis="y", alpha=0.3)

# ---------------------------------------------------------------------------
# 4. Legend and title
# ---------------------------------------------------------------------------
legend_elements = [Patch(facecolor=colors[c], alpha=0.75, label=c) for c in cases]
fig.legend(
    handles=legend_elements,
    loc="upper center",
    ncol=2,
    bbox_to_anchor=(0.5, 1.0),
    fontsize=11,
    frameon=False,
)
fig.suptitle(
    "Distribution of Accuracy, Params and FLOPs per dataset\n"
    "(all models vs ReLU/SELU models with < 300k parameters)",
    fontsize=14,
    y=1.1,
)

plt.tight_layout(rect=[0, 0, 1, 0.93])
plt.savefig(OUTPUT_PATH, dpi=150, bbox_inches="tight")
print(f"Chart saved to {OUTPUT_PATH}")