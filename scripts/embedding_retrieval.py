import json
from pathlib import Path

import numpy as np
from sentence_transformers import SentenceTransformer, util

# Reuse evaluation utilities from rule-based evaluation
from eval_rule_baselines import (
    build_label_matrix,
    evaluate_per_question,
    collect_examples_for_baseline
)
from rule_based import load_data


BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
RESULTS_DIR = BASE_DIR / "results"
RESULTS_DIR.mkdir(exist_ok=True)


def main():
    print("Loading labeled data...")
    clabels, qtexts, ctexts, q_ids, chunk_ids = load_data(DATA_DIR)

    # Build dictionary: (question_id, chunk_id) -> label
    labels_dict = {(q, c): lab for (q, c, lab) in clabels}

    # Convert into per-question label matrix (same shape as score matrix)
    per_question_labels = build_label_matrix(q_ids, chunk_ids, labels_dict)

    # -----------------------------------------------------------
    # 1. Load embedding model
    # -----------------------------------------------------------
    print("Loading embedding model...")
    model = SentenceTransformer("all-MiniLM-L6-v2")

    # -----------------------------------------------------------
    # 2. Encode questions and chunks
    # -----------------------------------------------------------
    print("Encoding questions...")
    Q = model.encode(qtexts, convert_to_tensor=True, show_progress_bar=True)

    print("Encoding chunks...")
    C = model.encode(ctexts, convert_to_tensor=True, show_progress_bar=True)

    # -----------------------------------------------------------
    # 3. Compute cosine similarity matrix
    # -----------------------------------------------------------
    print("Computing cosine similarities...")
    sim = util.cos_sim(Q, C)   # shape: (num_questions, num_chunks)

    # Convert from tensor to numpy
    scores = sim.cpu().numpy()

    # -----------------------------------------------------------
    # 4. Evaluate metrics
    # -----------------------------------------------------------
    print("Evaluating metrics...")
    metrics = evaluate_per_question(scores, per_question_labels, top_k=5)

    metrics_path = RESULTS_DIR / "embedding_metrics.json"
    with metrics_path.open("w", encoding="utf-8") as f:
        json.dump({"embedding": metrics}, f, indent=2, ensure_ascii=False)

    print(f"Saved metrics to {metrics_path}")

    # -----------------------------------------------------------
    # 5. Collect qualitative examples
    # -----------------------------------------------------------
    print("Collecting qualitative examples...")
    examples = collect_examples_for_baseline(
        scores,
        baseline_name="embedding",
        q_ids=q_ids,
        qtexts=qtexts,
        chunk_ids=chunk_ids,
        ctexts=ctexts,
        labels_dict=labels_dict,
        top_k=5,
        num_success=3,
        num_failure=3
    )

    examples_path = RESULTS_DIR / "embedding_examples.json"
    with examples_path.open("w", encoding="utf-8") as f:
        json.dump({"embedding": examples}, f, indent=2, ensure_ascii=False)

    print(f"Saved qualitative examples to {examples_path}")
    print("Done.")


if __name__ == "__main__":
    main()
