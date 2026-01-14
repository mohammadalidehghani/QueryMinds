import argparse
import csv
import json
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent

# Defaults for Task 3 (v2)
DEFAULT_QUESTIONS = BASE_DIR / "data" / "questions_v2.json"
DEFAULT_CANDIDATES = BASE_DIR / "data" / "candidate_chunks_v2.jsonl"
DEFAULT_OUT = BASE_DIR / "data" / "labels_blank_15x20_v2.csv"

DEFAULT_MAX_QUESTIONS = 15
DEFAULT_MAX_PER_QUESTION = 20


def _pick(d: dict, keys: list[str], default=None):
    """Return first existing key from keys in dict d."""
    for k in keys:
        if k in d and d[k] is not None:
            return d[k]
    return default


def load_questions(questions_path: Path):
    """
    Supports common formats:
    - [{"id": "...", "question": "..."}, ...]
    - [{"question_id": "...", "text": "..."}, ...]
    """
    with questions_path.open("r", encoding="utf-8") as f:
        data = json.load(f)

    if not isinstance(data, list):
        raise ValueError("Questions file must be a JSON list (array).")

    q_order = []
    q_text = {}

    for item in data:
        if not isinstance(item, dict):
            continue
        qid = _pick(item, ["id", "question_id", "qid"])
        q = _pick(item, ["question", "question_text", "text"], "")
        if qid is None:
            continue
        qid = str(qid)
        if qid not in q_text:
            q_order.append(qid)
        q_text[qid] = str(q)

    if not q_order:
        raise ValueError("Could not find question ids in the questions JSON.")

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
                raise ValueError(f"Invalid JSON on line {line_no} in {path}: {e}") from e


def main():
    ap = argparse.ArgumentParser(
        description="Create a blank 15×20 labeling CSV from a candidates JSONL file."
    )
    ap.add_argument("--questions", type=str, default=str(DEFAULT_QUESTIONS),
                    help="Path to questions JSON (default: data/questions2.json)")
    ap.add_argument("--candidates", type=str, default=str(DEFAULT_CANDIDATES),
                    help="Path to candidate chunks JSONL (default: data/candidate_chunks_v2.jsonl)")
    ap.add_argument("--out", type=str, default=str(DEFAULT_OUT),
                    help="Output CSV path (default: data/labels_blank_15x20_v2.csv)")
    ap.add_argument("--max_questions", type=int, default=DEFAULT_MAX_QUESTIONS,
                    help="How many questions to include (default: 15)")
    ap.add_argument("--max_per_question", type=int, default=DEFAULT_MAX_PER_QUESTION,
                    help="How many candidates per question (default: 20)")

    args = ap.parse_args()

    QUESTIONS_PATH = Path(args.questions)
    CANDIDATES_PATH = Path(args.candidates)
    OUT_PATH = Path(args.out)

    if not QUESTIONS_PATH.exists():
        raise FileNotFoundError(f"Missing questions file: {QUESTIONS_PATH}")
    if not CANDIDATES_PATH.exists():
        raise FileNotFoundError(f"Missing candidates file: {CANDIDATES_PATH}")

    q_order, q_text = load_questions(QUESTIONS_PATH)

    selected_qids = q_order[: args.max_questions]
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

        qid = str(qid)
        cid = str(cid)

        if qid not in selected_set:
            continue
        if per_q_count[qid] >= args.max_per_question:
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
            "chunk_text": str(ctext),

            "person_1": "",
            "person_2": "",
            "person_3": "",
            "final_label": "",
        })

        if all(per_q_count[x] >= args.max_per_question for x in selected_qids):
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
    print(f"Questions file: {QUESTIONS_PATH}")
    print(f"Candidates file: {CANDIDATES_PATH}")
    print("Counts per question:")
    for qid in selected_qids:
        print(f"  {qid}: {per_q_count[qid]}/{args.max_per_question}")
    print(f"Total rows: {len(rows)} (max = {args.max_questions * args.max_per_question})")

    if len(rows) < args.max_questions * args.max_per_question:
        print("WARNING: Not enough candidates to fill full 15×20 for all questions.")


if __name__ == "__main__":
    main()
