import json
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
RESULTS_DIR = PROJECT_ROOT / "results"
DOCS_DIR = PROJECT_ROOT / "docs"
DOCS_DIR.mkdir(exist_ok=True)


def load_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def extract_metrics(baseline_name, metrics_json):
    """Extract metrics for a baseline from its JSON dictionary."""
    d = metrics_json[baseline_name]

    return {
        "Baseline": baseline_name,
        "Precision": d.get("precision"),
        "Recall": d.get("recall"),
        "F1": d.get("f1"),
        "Accuracy": d.get("accuracy"),
        "Precision@k": d.get("precision_at_k"),
        "Recall@k": d.get("recall_at_k"),
    }


def main():

    rows = []

    # RULE BASED (2 baselines)
    rule_json = load_json(RESULTS_DIR / "rule_based_metrics.json")
    rows.append(extract_metrics("keyword_overlap", rule_json))
    rows.append(extract_metrics("tfidf_cosine", rule_json))

    # EMBEDDING
    emb_json = load_json(RESULTS_DIR / "embedding_metrics.json")
    rows.append(extract_metrics("embedding", emb_json))

    # SUPERVISED
    sup_json = load_json(RESULTS_DIR / "supervised_classifier_metrics.json")
    rows.append(extract_metrics("supervised_classifier", sup_json))

    # DataFrame
    df = pd.DataFrame(rows)
    baselines = df["Baseline"]

    # -------------------------
    # PLOT 1 – Precision / Recall / F1
    # -------------------------
    plt.figure(figsize=(10, 6))
    x = np.arange(len(df))

    plt.bar(x - 0.25, df["Precision"], width=0.25, label="Precision")
    plt.bar(x, df["Recall"], width=0.25, label="Recall")
    plt.bar(x + 0.25, df["F1"], width=0.25, label="F1")

    plt.xticks(x, baselines, rotation=15)
    plt.title("Baseline Comparison: Precision, Recall, F1")
    plt.ylabel("Score")
    plt.legend()
    plt.tight_layout()
    plt.savefig(DOCS_DIR / "plot_prf.png", dpi=200)
    plt.close()

    # -------------------------
    # PLOT 2 – Precision@k / Recall@k
    # -------------------------
    plt.figure(figsize=(10, 6))

    plt.bar(x - 0.15, df["Precision@k"], width=0.3, label="Precision@k")
    plt.bar(x + 0.15, df["Recall@k"], width=0.3, label="Recall@k")

    plt.xticks(x, baselines, rotation=15)
    plt.title("Top-k Retrieval Metrics")
    plt.ylabel("Score")
    plt.legend()
    plt.tight_layout()
    plt.savefig(DOCS_DIR / "plot_atk.png", dpi=200)
    plt.close()


if __name__ == "__main__":
    main()
