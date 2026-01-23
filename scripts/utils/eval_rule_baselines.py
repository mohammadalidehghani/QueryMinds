"""
Evaluate rule-based baselines (keyword overlap + TF-IDF cosine)
and generate qualitative examples (successes / failures).

Usage (from project root):
    python scripts/eval_rule_baselines.py
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Tuple
import numpy as np
from sklearn.metrics import precision_recall_fscore_support, accuracy_score
import sys
sys.dont_write_bytecode = True

SCRIPT_DIR = Path(__file__).resolve().parent.parent
PROJECT_ROOT = SCRIPT_DIR.parent.parent
BASELINES_DIR = PROJECT_ROOT / "baselines"
sys.path.append(str(BASELINES_DIR))

DATA_DIR = PROJECT_ROOT / "data"
RESULTS_DIR = PROJECT_ROOT / "results"
RESULTS_DIR.mkdir(exist_ok=True)


from rule_based import (
    load_data,
    keyword_overlap_scores,
    tfidf_cosine_scores,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"
RESULTS_DIR = PROJECT_ROOT / "results"
RESULTS_DIR.mkdir(exist_ok=True)


def build_label_matrix(
    q_ids: List[str],
    chunk_ids: List[str],
    labels_dict: Dict[Tuple[str, str], int],
):


    per_question = {}
    for q in q_ids:
        y = []
        m = []
        for c in chunk_ids:
            key = (q, c)
            if key in labels_dict:
                y.append(labels_dict[key])
                m.append(True)
            else:

                y.append(0)
                m.append(False)
        per_question[q] = (np.array(y), np.array(m))
    return per_question


# ---------------------------------------------------------------------------
#   (metrics)
# ---------------------------------------------------------------------------

def evaluate_per_question(
    scores: np.ndarray,
    per_question_labels,
    top_k: int = 5,
):


    q_ids = list(per_question_labels.keys())
    all_prec, all_rec, all_f1, all_acc = [], [], [], []
    all_prec_at_k, all_rec_at_k = [], []

    for i, q_id in enumerate(q_ids):
        y_true, mask = per_question_labels[q_id]

        y_true = y_true[mask]
        y_scores = scores[i][mask]

        if y_true.size == 0:

            continue


        y_pred = (y_scores > 0).astype(int)

        prec, rec, f1, _ = precision_recall_fscore_support(
            y_true, y_pred, average="binary", zero_division=0
        )
        acc = accuracy_score(y_true, y_pred)

        all_prec.append(prec)
        all_rec.append(rec)
        all_f1.append(f1)
        all_acc.append(acc)


        order = np.argsort(-y_scores)
        k = min(top_k, len(order))
        top_idx = order[:k]

        relevant = y_true == 1
        retrieved_relevant = relevant[top_idx].sum()
        total_relevant = relevant.sum()

        prec_k = retrieved_relevant / k if k > 0 else 0.0
        rec_k = retrieved_relevant / total_relevant if total_relevant > 0 else 0.0

        all_prec_at_k.append(prec_k)
        all_rec_at_k.append(rec_k)

    metrics = {
        "precision": float(np.mean(all_prec)) if all_prec else 0.0,
        "recall": float(np.mean(all_rec)) if all_rec else 0.0,
        "f1": float(np.mean(all_f1)) if all_f1 else 0.0,
        "accuracy": float(np.mean(all_acc)) if all_acc else 0.0,
        "precision_at_k": float(np.mean(all_prec_at_k)) if all_prec_at_k else 0.0,
        "recall_at_k": float(np.mean(all_rec_at_k)) if all_rec_at_k else 0.0,
        "top_k": top_k,
        "num_questions": len(q_ids),
    }
    return metrics



def _build_example_dict(
    q_idx: int,
    q_id: str,
    qtexts: List[str],
    chunk_ids: List[str],
    ctexts: List[str],
    labeled_sorted: List[Tuple[int, str, int, float]],
    top_k: int,
    baseline_name: str,
):

    k = min(top_k, len(labeled_sorted))
    top = labeled_sorted[:k]

    example = {
        "baseline": baseline_name,
        "question_id": q_id,
        "question_text": qtexts[q_idx],
        "top_k": [],
    }

    for rank, (c_idx, c_id, lab, sc) in enumerate(top, start=1):
        example["top_k"].append(
            {
                "rank": rank,
                "chunk_id": c_id,

                "chunk_text_preview": ctexts[c_idx][:400],
                "label": int(lab),
                "score": float(sc),
            }
        )

    return example


def collect_examples_for_baseline(
    scores: np.ndarray,
    baseline_name: str,
    q_ids: List[str],
    qtexts: List[str],
    chunk_ids: List[str],
    ctexts: List[str],
    labels_dict: Dict[Tuple[str, str], int],
    top_k: int = 5,
    num_success: int = 3,
    num_failure: int = 3,
):


    success_examples = []
    failure_examples = []

    for qi, q_id in enumerate(q_ids):

        labeled = []
        for cj, c_id in enumerate(chunk_ids):
            key = (q_id, c_id)
            if key in labels_dict:
                lab = labels_dict[key]
                sc = float(scores[qi, cj])
                labeled.append((cj, c_id, lab, sc))

        if not labeled:
            continue


        if not any(lab == 1 for (_, _, lab, _) in labeled):
            continue


        labeled_sorted = sorted(labeled, key=lambda x: x[3], reverse=True)

        top1_label = labeled_sorted[0][2]


        if top1_label == 1 and len(success_examples) < num_success:
            ex = _build_example_dict(
                qi,
                q_id,
                qtexts,
                chunk_ids,
                ctexts,
                labeled_sorted,
                top_k,
                baseline_name,
            )
            success_examples.append(ex)
            continue


        if top1_label == 0 and len(failure_examples) < num_failure:
            ex = _build_example_dict(
                qi,
                q_id,
                qtexts,
                chunk_ids,
                ctexts,
                labeled_sorted,
                top_k,
                baseline_name,
            )
            failure_examples.append(ex)

        if len(success_examples) >= num_success and len(failure_examples) >= num_failure:
            break

    return {
        "success_examples": success_examples,
        "failure_examples": failure_examples,
        "top_k": top_k,
    }


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------

def main():
    print("Loading labeled data...")
    clabels, qtexts, ctexts, q_ids, chunk_ids = load_data(DATA_DIR)


    labels_dict = {(q, c): lab for (q, c, lab) in clabels}
    per_question_labels = build_label_matrix(q_ids, chunk_ids, labels_dict)

    # ---------- baseline 1: keyword overlap ----------
    print("Computing keyword overlap scores...")
    kw_scores = keyword_overlap_scores(qtexts, ctexts, DATA_DIR)
    kw_metrics = evaluate_per_question(kw_scores, per_question_labels, top_k=5)
    print("Keyword overlap metrics:", kw_metrics)

    # ---------- baseline 2: TF-IDF cosine ----------
    print("Computing TF-IDF cosine scores...")
    tfidf_scores = tfidf_cosine_scores(qtexts, ctexts)
    tfidf_metrics = evaluate_per_question(tfidf_scores, per_question_labels, top_k=5)
    print("TF-IDF cosine metrics:", tfidf_metrics)


    metrics_path = RESULTS_DIR / "rule_based_metrics.json"
    with metrics_path.open("w", encoding="utf-8") as f:
        json.dump(
            {
                "keyword_overlap": kw_metrics,
                "tfidf_cosine": tfidf_metrics,
            },
            f,
            indent=2,
            ensure_ascii=False,
        )

    print(f"\nSaved metrics to {metrics_path}")


    print("\nCollecting qualitative examples (success/failure) ...")

    kw_examples = collect_examples_for_baseline(
        kw_scores,
        baseline_name="keyword_overlap",
        q_ids=q_ids,
        qtexts=qtexts,
        chunk_ids=chunk_ids,
        ctexts=ctexts,
        labels_dict=labels_dict,
        top_k=5,
        num_success=3,
        num_failure=3,
    )

    tfidf_examples = collect_examples_for_baseline(
        tfidf_scores,
        baseline_name="tfidf_cosine",
        q_ids=q_ids,
        qtexts=qtexts,
        chunk_ids=chunk_ids,
        ctexts=ctexts,
        labels_dict=labels_dict,
        top_k=5,
        num_success=3,
        num_failure=3,
    )

    examples_path = RESULTS_DIR / "rule_based_examples.json"
    with examples_path.open("w", encoding="utf-8") as f:
        json.dump(
            {
                "keyword_overlap": kw_examples,
                "tfidf_cosine": tfidf_examples,
            },
            f,
            indent=2,
            ensure_ascii=False,
        )

    print(f"Saved qualitative examples to {examples_path}\n")


if __name__ == "__main__":
    main()
