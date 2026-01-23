import csv
import json
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent

QUESTIONS_PATH = BASE_DIR / "data" / "questions_v2.json"
CANDIDATES_PATH = BASE_DIR / "data" / "chunks" / "candidate_chunks.jsonl"
OUT_PATH = BASE_DIR / "data" / "labels" / "labels_blank_15x20.csv"

MAX_QUESTIONS = 15
MAX_PER_QUESTION = 20

def _pick(d: dict, keys: list[str], default=None):
    """Return first existing key from keys in dict d."""
    for k in keys:
        if k in d and d[k] is not None:
            return d[k]
    return default


def load_questions():
    """
    Supports common formats:
    - [{"id": "...", "question": "..."}, ...]
    - [{"question_id": "...", "text": "..."}, ...]
    """
    with QUESTIONS_PATH.open("r", encoding="utf-8") as f:
        data = json.load(f)

    if not isinstance(data, list):
        raise ValueError("questions.json must be a JSON list.")

    q_order = []
    q_text = {}

    for item in data:
        if not isinstance(item, dict):
            continue
        qid = _pick(item, ["id", "question_id", "qid"])
        q = _pick(item, ["question", "question_text", "text"], "")
        if qid is None:
            continue
        if qid not in q_text:
            q_order.append(qid)
        q_text[qid] = q

    if not q_order:
        raise ValueError("Could not find question ids in questions.json.")

    return q_order, q_text


def iter_jsonl(path: Path):
    with path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError as e:
                raise ValueError(f"Invalid JSON on line {line_no} in {path}: {e}")


def main():
    if not QUESTIONS_PATH.exists():
        raise FileNotFoundError(f"Missing: {QUESTIONS_PATH}")
    if not CANDIDATES_PATH.exists():
        raise FileNotFoundError(f"Missing: {CANDIDATES_PATH}")

    q_order, q_text = load_questions()
    selected_qids = q_order[:MAX_QUESTIONS]
    selected_set = set(selected_qids)

    per_q_count = {qid: 0 for qid in selected_qids}
    seen_pairs = set()

    rows = []

    for rec in iter_jsonl(CANDIDATES_PATH):
        if not isinstance(rec, dict):
            continue

        qid = _pick(rec, ["question_id", "qid", "questionId", "question"])
        cid = _pick(rec, ["chunk_id", "cid", "chunkId", "chunk"])
        ctext = _pick(rec, ["text", "chunk_text", "content"], "")

        if qid is None or cid is None:
            continue
        if qid not in selected_set:
            continue
        if per_q_count[qid] >= MAX_PER_QUESTION:
            continue

        pair = (qid, cid)
        if pair in seen_pairs:
            continue
        seen_pairs.add(pair)

        per_q_count[qid] += 1
        rows.append({
            "question_id": qid,
            "chunk_id": cid,
            "question_text": q_text.get(qid, ""),
            "chunk_text": ctext,

            "person_1": "",
            "person_2": "",
            "person_3": "",
            "final_label": "",
        })

        if all(per_q_count[x] >= MAX_PER_QUESTION for x in selected_qids):
            break

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "question_id", "chunk_id", "question_text", "chunk_text",
        "person_1", "person_2", "person_3", "final_label"
    ]
    with OUT_PATH.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print(f"Created: {OUT_PATH}")
    print("Counts per question:")
    for qid in selected_qids:
        print(f"  {qid}: {per_q_count[qid]}/{MAX_PER_QUESTION}")
    print(f"Total rows: {len(rows)} (max = {MAX_QUESTIONS*MAX_PER_QUESTION})")

    if len(rows) < MAX_QUESTIONS * MAX_PER_QUESTION:
        print("WARNING: Not enough candidates to fill full 15×20 for all questions.")


if __name__ == "__main__":
    main()
