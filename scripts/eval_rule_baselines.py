"""
Evaluate rule-based baselines (keyword overlap + TF-IDF cosine).

Usage (from project root):
    python scripts/eval_rule_baselines.py
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import precision_recall_fscore_support, accuracy_score

# چون فایل‌های ما کنار همین اسکریپت هستند:
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
    """
    بر اساس question_id و chunk_id و دیکشنری لیبل‌ها،
    یک لیست از (y_true, mask) می‌سازد برای هر سوال.
    """
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


def evaluate_per_question(
    scores: np.ndarray,
    per_question_labels,
    top_k: int = 5,
):
    """
    برای هر سوال، با استفاده از نمره‌ها و لیبل‌های حقیقی، متریک‌ها را حساب می‌کند.
    scores: آرایه با شکل (num_questions, num_chunks)
    per_question_labels: خروجی build_label_matrix
    """
    q_ids = list(per_question_labels.keys())
    all_prec, all_rec, all_f1, all_acc = [], [], [], []
    all_prec_at_k, all_rec_at_k = [], []

    for i, q_id in enumerate(q_ids):
        y_true, mask = per_question_labels[q_id]
        # فقط نمونه‌هایی که لیبل واقعی دارند
        y_true = y_true[mask]
        y_scores = scores[i][mask]

        # آستانه صفر برای باینری‌کردن
        y_pred = (y_scores > 0).astype(int)

        prec, rec, f1, _ = precision_recall_fscore_support(
            y_true, y_pred, average="binary", zero_division=0
        )
        acc = accuracy_score(y_true, y_pred)

        all_prec.append(prec)
        all_rec.append(rec)
        all_f1.append(f1)
        all_acc.append(acc)

        # Precision@k / Recall@k بر اساس مرتب‌سازی نزولی نمره‌ها
        # اگر تعداد label=1 کم باشد، ممکن است k را adjust کنیم
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
        "precision": float(np.mean(all_prec)),
        "recall": float(np.mean(all_rec)),
        "f1": float(np.mean(all_f1)),
        "accuracy": float(np.mean(all_acc)),
        "precision_at_k": float(np.mean(all_prec_at_k)),
        "recall_at_k": float(np.mean(all_rec_at_k)),
        "top_k": top_k,
        "num_questions": len(q_ids),
    }
    return metrics


def main():
    print("Loading labeled data...")
    clabels, qtexts, ctexts, q_ids, chunk_ids = load_data(DATA_DIR)

    # تبدیل لیبل‌ها به دیکشنری برای دسترسی سریع
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

    # ذخیره نتایج
    out_path = RESULTS_DIR / "rule_based_metrics.json"
    with out_path.open("w", encoding="utf-8") as f:
        json.dump(
            {
                "keyword_overlap": kw_metrics,
                "tfidf_cosine": tfidf_metrics,
            },
            f,
            indent=2,
            ensure_ascii=False,
        )

    print(f"\nSaved metrics to {out_path}")


if __name__ == "__main__":
    main()
