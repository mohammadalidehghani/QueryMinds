import json
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent

QUESTIONS_PATH = BASE_DIR / "data" / "questions.json"
CANDIDATES_PATH = BASE_DIR / "data" / "candidate_chunks.jsonl"
GOLD_PATH = BASE_DIR / "data" / "gold_labels.jsonl"

# ---- YOU CAN ADJUST THESE LIMITS ----
MAX_TOTAL_LABELS = 274      # total labels you want in the end (None = no limit) 
MAX_PER_QUESTION = 20       # max labels per question (None = no limit) 
# -------------------------------------


def load_questions():
    with QUESTIONS_PATH.open("r", encoding="utf-8") as f:
        questions = json.load(f)
    return {q["id"]: q["question"] for q in questions}


def iter_candidates():
    with CANDIDATES_PATH.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            yield json.loads(line)


def load_existing_labels():
    """Return (set of (qid, cid), dict of counts per question)."""
    labeled = set()
    per_q = {}
    if GOLD_PATH.exists():
        with GOLD_PATH.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    obj = json.loads(line)
                except json.JSONDecodeError:
                    continue
                qid = obj.get("question_id")
                cid = obj.get("chunk_id")
                if qid is None or cid is None:
                    continue
                pair = (qid, cid)
                if pair in labeled:
                    continue
                labeled.add(pair)
                per_q[qid] = per_q.get(qid, 0) + 1
    return labeled, per_q


def main():
    questions = load_questions()
    print(f"Loaded {len(questions)} questions.")

    labeled_pairs, per_q = load_existing_labels()
    total_labeled = len(labeled_pairs)
    print(f"Found {total_labeled} already labeled pairs in {GOLD_PATH.name}.")

    if MAX_TOTAL_LABELS is not None and total_labeled >= MAX_TOTAL_LABELS:
        print(f"Already reached MAX_TOTAL_LABELS = {MAX_TOTAL_LABELS}. Nothing to do.")
        return

    with GOLD_PATH.open("a", encoding="utf-8") as out_f:
        for rec in iter_candidates():
            qid = rec["question_id"]
            cid = rec["chunk_id"]

            # Skip if already labeled
            if (qid, cid) in labeled_pairs:
                continue

            # Skip if this question already has enough labels
            if MAX_PER_QUESTION is not None and per_q.get(qid, 0) >= MAX_PER_QUESTION:
                continue

            # Stop if we hit total label limit
            if MAX_TOTAL_LABELS is not None and total_labeled >= MAX_TOTAL_LABELS:
                print(f"\nReached MAX_TOTAL_LABELS = {MAX_TOTAL_LABELS}. Stopping.")
                break

            question_text = questions.get(qid, f"[UNKNOWN QUESTION {qid}]")
            text = rec["text"]
            matched = rec.get("matched_keywords", [])
            section = rec.get("section")

            print("\n" + "=" * 80)
            print(f"Question {qid}: {question_text}")
            print(f"Chunk: {cid}  (section: {section})")
            if matched:
                print(f"Matched keywords: {', '.join(matched)}")
            print("-" * 80)
            print(text)
            print("-" * 80)
            print(f"(Already labeled: total={total_labeled}, q={qid} -> {per_q.get(qid, 0)})")
            ans = input("Label this chunk? [1=relevant, 0=not, s=skip, q=quit] > ").strip()

            if ans.lower() == "q":
                print("Stopping labeling on user request.")
                break
            if ans.lower() == "s" or ans == "":
                print("Skipped (will appear again next time).")
                continue
            if ans not in {"0", "1"}:
                print("Invalid input, skipping (will appear again next time).")
                continue

            label = int(ans)
            out_obj = {
                "question_id": qid,
                "chunk_id": cid,
                "label": label,
            }
            out_f.write(json.dumps(out_obj, ensure_ascii=False) + "\n")
            out_f.flush()

            # Update bookkeeping
            labeled_pairs.add((qid, cid))
            total_labeled += 1
            per_q[qid] = per_q.get(qid, 0) + 1

            print(f"Saved label {label} for ({qid}, {cid}). "
                  f"Now: total={total_labeled}, q={qid}->{per_q[qid]}")

    print(f"\nDone for now. Labels are stored in {GOLD_PATH}")


if __name__ == "__main__":
    main()
