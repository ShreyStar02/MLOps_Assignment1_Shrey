"""Exploratory data analysis: writes figures to reports/eda and a summary JSON."""
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import seaborn as sns  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.data import CATEGORICAL, NUMERIC, TARGET, load_clean  # noqa: E402

OUT = ROOT / "reports" / "eda"


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    df = load_clean(ROOT / "data" / "processed" / "heart_clean.csv")
    sns.set_theme(style="whitegrid")

    ax = df[TARGET].value_counts().sort_index().plot(
        kind="bar", color=["#4c9f70", "#d1495b"], rot=0)
    ax.set_xticklabels(["No disease (0)", "Disease (1)"])
    ax.set_title("Class balance")
    ax.figure.savefig(OUT / "class_balance.png", dpi=150, bbox_inches="tight")
    plt.close(ax.figure)

    fig, axes = plt.subplots(2, 3, figsize=(14, 7))
    for a, col in zip(axes.ravel(), NUMERIC):
        sns.histplot(data=df, x=col, hue=TARGET, kde=True, ax=a, palette="Set1")
    axes.ravel()[-1].axis("off")
    fig.suptitle("Numeric feature distributions by target")
    fig.savefig(OUT / "histograms.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(df.corr(), annot=True, fmt=".2f", cmap="coolwarm", center=0, ax=ax)
    ax.set_title("Correlation heatmap")
    fig.savefig(OUT / "correlation_heatmap.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    fig, axes = plt.subplots(2, 4, figsize=(16, 7))
    for a, col in zip(axes.ravel(), CATEGORICAL):
        sns.countplot(data=df, x=col, hue=TARGET, ax=a, palette="Set1")
    fig.suptitle("Categorical features vs target")
    fig.savefig(OUT / "categorical_vs_target.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    fig, axes = plt.subplots(1, 5, figsize=(16, 4))
    for a, col in zip(axes, NUMERIC):
        sns.boxplot(data=df, x=TARGET, y=col, hue=TARGET, ax=a, palette="Set2", legend=False)
    fig.savefig(OUT / "boxplots.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    summary = {
        "rows": len(df),
        "missing_values": df.isna().sum()[lambda s: s > 0].to_dict(),
        "class_counts": df[TARGET].value_counts().to_dict(),
        "describe": df[NUMERIC].describe().round(2).to_dict(),
    }
    (OUT / "summary.json").write_text(json.dumps(summary, indent=2, default=int))
    print(json.dumps(summary["class_counts"], default=int), summary["missing_values"])


if __name__ == "__main__":
    main()
