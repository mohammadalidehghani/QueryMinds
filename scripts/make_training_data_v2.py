import json
from pathlib import Path

# Paths
BASE_DIR = Path(__file__).resolve().parent.parent
QUESTIONS_PATH = BASE_DIR / "data" / "questions_v2.json"
CHUNKS_PATH = BASE_DIR / "data" / "chunks_30_v2.jsonl"
GOLD_PATH = BASE_DIR / "data" / "gold_labels_v2.jsonl"

OUT_PATH = BASE_DIR / "data" / "ml_pairs_v2.jsonl"


def load_questions():
    with QUESTIONS_PATH.open("r", encoding="utf-8") as f:
        data = json.load(f)
    # Map id -> question text
    return {q["id"]: q["question"] for q in data}


def load_chunks():
    chunk_map = {}
    with CHUNKS_PATH.open("r", encoding="utf-8") as f:
        for line in f:
            obj = json.loads(line)
            chunk_map[obj["chunk_id"]] = obj["text"]
    return chunk_map


def load_gold_labels():
    """Return list of (question_id, chunk_id, label)."""
    labels = []
    with GOLD_PATH.open("r", encoding="utf-8") as f:
        for line in f:
            obj = json.loads(line)
            labels.append((obj["question_id"], obj["chunk_id"], int(obj["label"])))
    return labels


def main():
    print("Loading data...")

    q_map = load_questions()
    c_map = load_chunks()
    gold = load_gold_labels()

    print(f"Questions: {len(q_map)}")
    print(f"Chunks: {len(c_map)}")
    print(f"Gold labels: {len(gold)}")

    OUT_PATH.parent.mkdir(exist_ok=True)

    # Build pairs
    with OUT_PATH.open("w", encoding="utf-8") as out_f:
        for qid, cid, lab in gold:
            if qid not in q_map or cid not in c_map:
                continue

            record = {
                "question_id": qid,
                "question": q_map[qid],
                "chunk_id": cid,
                "chunk": c_map[cid],
                "label": lab,
            }
            out_f.write(json.dumps(record, ensure_ascii=False) + "\n")

    print(f"Saved training pairs to {OUT_PATH}")


if __name__ == "__main__":
    main()
