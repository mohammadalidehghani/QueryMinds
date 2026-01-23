import json
from pathlib import Path

# Base project folder: .../QueryMinds-main
BASE_DIR = Path(__file__).resolve().parent.parent

# Input files
QUESTIONS_PATH = BASE_DIR / "data" / "questions.json"
CHUNKS_PATH = BASE_DIR / "data" / "chunks" / "chunks_30.jsonl"   # use your subset file

# Output file
OUT_PATH = BASE_DIR / "data" / "chunks" / "candidate_chunks.jsonl"

# Optional: limit how many candidates per question (set to None for no limit)
MAX_PER_QUESTION = None  # e.g. set to 50 if you want max 50 per question

def load_questions():
    with QUESTIONS_PATH.open("r", encoding="utf-8") as f:
        questions = json.load(f)
    # Normalize keywords to lowercase
    for q in questions:
        q["keywords"] = [kw.lower() for kw in q.get("keywords", [])]
    return questions

def iter_chunks():
    with CHUNKS_PATH.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            # add lowercase text for matching
            obj["_text_lower"] = obj["text"].lower()
            yield obj

def find_candidates_for_question(question, chunks):
    keywords = question["keywords"]
    q_id = question["id"]

    candidates = []
    for chunk in chunks:
        text_lower = chunk["_text_lower"]
        matched = [kw for kw in keywords if kw in text_lower]
        if matched:
            record = {
                "question_id": q_id,
                "chunk_id": chunk["chunk_id"],
                "paper_id": chunk["paper_id"],
                "section": chunk["section"],
                "text": chunk["text"],
                "matched_keywords": matched,
            }
            candidates.append(record)

    # Optionally limit number per question
    if MAX_PER_QUESTION is not None and len(candidates) > MAX_PER_QUESTION:
        candidates = candidates[:MAX_PER_QUESTION]

    return candidates

def main():
    print(f"Loading questions from {QUESTIONS_PATH} ...")
    questions = load_questions()
    print(f"Found {len(questions)} questions.")

    # Load chunks once into memory (205 chunks is small)
    print(f"Loading chunks from {CHUNKS_PATH} ...")
    chunks = list(iter_chunks())
    print(f"Found {len(chunks)} chunks.")

    total_records = 0
    with OUT_PATH.open("w", encoding="utf-8") as out_f:
        for q in questions:
            q_id = q["id"]
            print(f"\nFinding candidates for {q_id}: {q['question']}")
            candidates = find_candidates_for_question(q, chunks)
            print(f"  -> {len(candidates)} candidate chunks")

            for rec in candidates:
                out_f.write(json.dumps(rec, ensure_ascii=False) + "\n")
                total_records += 1

    print(f"\nDone. Wrote {total_records} records to {OUT_PATH}")

if __name__ == "__main__":
    main()
