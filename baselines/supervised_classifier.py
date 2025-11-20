import json
from pathlib import Path

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
import sys

SCRIPT_DIR = Path(__file__).resolve().parent         
PROJECT_ROOT = SCRIPT_DIR.parent                   
DATA_DIR = PROJECT_ROOT / "data"
RESULTS_DIR = PROJECT_ROOT / "results"
RESULTS_DIR.mkdir(exist_ok=True)
BASELINES_DIR = PROJECT_ROOT / "baselines"
SCRIPTS_DIR = PROJECT_ROOT / "scripts"
DOCS_DIR = PROJECT_ROOT / "docs"
sys.path.append(str(BASELINES_DIR))
sys.path.append(str(SCRIPTS_DIR))


from eval_rule_baselines import (
    build_label_matrix,
    evaluate_per_question,
    collect_examples_for_baseline
)
from rule_based import load_data




def load_ml_pairs():
    path = DATA_DIR / "ml_pairs.jsonl"
    pairs = []

    with path.open("r", encoding="utf-8") as f:
        for line in f:
            obj = json.loads(line)
            pairs.append(obj)

    return pairs


def main():
    print("Loading training pairs...")
    pairs = load_ml_pairs()

    # Build texts + labels
    X = []
    y = []

    for p in pairs:
        # Concatenate question + chunk
        text = p["question"] + " [SEP] " + p["chunk"]
        X.append(text)
        y.append(p["label"])

    y = np.array(y)

    # ------------------------------------------------------------
    # Train / dev split
    # ------------------------------------------------------------
    X_train, X_dev, y_train, y_dev = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # ------------------------------------------------------------
    # TF-IDF vectorizer
    # ------------------------------------------------------------
    vectorizer = TfidfVectorizer(
        stop_words="english",
        max_features=50000
    )
    X_train_vec = vectorizer.fit_transform(X_train)
    X_dev_vec = vectorizer.transform(X_dev)

    # ------------------------------------------------------------
    # Classifier (Logistic Regression)
    # ------------------------------------------------------------
    clf = LogisticRegression(
        max_iter=5000,
        class_weight="balanced"
    )
    clf.fit(X_train_vec, y_train)

    # Dev predictions
    y_pred = clf.predict(X_dev_vec)

    # Base metrics
    acc = accuracy_score(y_dev, y_pred)
    prec, rec, f1, _ = precision_recall_fscore_support(
        y_dev, y_pred, average="binary", zero_division=0
    )

    # ------------------------------------------------------------
    # Now compute per-question top-k metrics
    # Using the same evaluation as rule-based & embedding baseline
    # ------------------------------------------------------------
    print("Loading data for question/chunk-level evaluation...")
    clabels, qtexts, ctexts, q_ids, chunk_ids = load_data(DATA_DIR)
    labels_dict = {(q, c): lab for (q, c, lab) in clabels}
    per_question_labels = build_label_matrix(q_ids, chunk_ids, labels_dict)

    # Predict score for each (q, c) pair
    print("Computing classifier scores for retrieval evaluation...")

    # Prepare all texts for scoring: question + chunk
    all_texts = []
    for q in qtexts:
        for c in ctexts:
            all_texts.append(q + " [SEP] " + c)

    all_vec = vectorizer.transform(all_texts)
    all_scores = clf.predict_proba(all_vec)[:, 1]   # probability of label=1
    all_scores = all_scores.reshape(len(qtexts), len(ctexts))

    # Evaluate retrieval metrics
    metrics = evaluate_per_question(all_scores, per_question_labels, top_k=5)

    # Add classifier-only metrics
    metrics["clf_accuracy_dev"] = acc
    metrics["clf_precision_dev"] = float(prec)
    metrics["clf_recall_dev"] = float(rec)
    metrics["clf_f1_dev"] = float(f1)

    # Save metrics
    metrics_path = RESULTS_DIR / "supervised_classifier_metrics.json"
    with metrics_path.open("w", encoding="utf-8") as f:
        json.dump({"supervised_classifier": metrics}, f, indent=2, ensure_ascii=False)

    print(f"Saved classifier metrics to {metrics_path}")
    



    # ------------------------------------------------------------
    # Qualitative examples
    # ------------------------------------------------------------
    examples = collect_examples_for_baseline(
        all_scores,
        baseline_name="supervised_classifier",
        q_ids=q_ids,
        qtexts=qtexts,
        chunk_ids=chunk_ids,
        ctexts=ctexts,
        labels_dict=labels_dict,
        top_k=5,
        num_success=3,
        num_failure=3
    )

    examples_path = RESULTS_DIR / "supervised_classifier_examples.json"
    with examples_path.open("w", encoding="utf-8") as f:
        json.dump({"supervised_classifier": examples}, f, indent=2, ensure_ascii=False)

    print(f"Saved qualitative examples to {examples_path}")



if __name__ == "__main__":
    main()
