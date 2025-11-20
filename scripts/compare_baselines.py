import json
from pathlib import Path
import pandas as pd

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
RESULTS_DIR = PROJECT_ROOT / "results"
DOCS_DIR = PROJECT_ROOT / "docs"
DOCS_DIR.mkdir(exist_ok=True)


def load_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def normalize_row(name, d):
    return {
        "Baseline": name,
        "Precision": d.get("precision"),
        "Recall": d.get("recall"),
        "F1": d.get("f1"),
        "Accuracy": d.get("accuracy"),
        "Precision@k": d.get("precision_at_k"),
        "Recall@k": d.get("recall_at_k"),
        "k": d.get("top_k"),
        "Num Questions": d.get("num_questions"),
    }


def main():
    rows = []

    # RULE-BASED (2 metode)
    rule_data = load_json(RESULTS_DIR / "rule_based_metrics.json")
    for baseline_name, metrics in rule_data.items():
        rows.append(normalize_row(baseline_name, metrics))

    # EMBEDDING
    emb_data = load_json(RESULTS_DIR / "embedding_metrics.json")
    for baseline_name, metrics in emb_data.items():
        rows.append(normalize_row(baseline_name, metrics))

    # SUPERVISED
    sup_data = load_json(RESULTS_DIR / "supervised_classifier_metrics.json")
    for baseline_name, metrics in sup_data.items():
        rows.append(normalize_row(baseline_name, metrics))

    df = pd.DataFrame(rows)

    # SAVE MARKDOWN ONLY
    md_path = DOCS_DIR / "baseline_comparison.md"
    with open(md_path, "w", encoding="utf-8") as f:
        f.write("# Baseline Comparison\n\n")
        f.write(df.to_markdown(index=False))


if __name__ == "__main__":
    main()
