"""
Task 3 — Improved chunking (v2)

What this script does (compared to build_chunks_30.py):
- Section-aware chunk sizing:
  * abstract: smaller chunks, smaller overlap
  * body/other sections: larger chunks, larger overlap
- Sentence-boundary overlap (reduces "contextually unclear" chunks that start mid-sentence)
- Optional skipping of low-value sections (title, references, etc.)
- Writes to a NEW output file: data/chunks_30_v2.jsonl (does not overwrite chunks_30.jsonl)

Input:
- data/selected_30/*.json

Output:
- data/chunks/chunks_30_v2.jsonl

Run:
  python scripts/build_chunks_30_v2.py
"""

import json
from pathlib import Path
from collections import OrderedDict
import re
from typing import Dict, List, Tuple, Iterable, Optional

# Project root
BASE_DIR = Path(__file__).resolve().parent.parent.parent

# Our 30 selected JSON files
IN_DIR = BASE_DIR / "data" / "selected_30"

# Where we save all chunks (one big JSONL file)
OUT_PATH = BASE_DIR / "data" / "chunks" / "chunks_30_v2.jsonl"

# Section-aware chunk settings (tune if needed)
MAX_TOKENS_ABSTRACT = 160
OVERLAP_ABSTRACT = 40

MAX_TOKENS_BODY = 240
OVERLAP_BODY = 60

# Sections that are usually not useful as standalone retrieval evidence
SKIP_SECTIONS = {
    "title",
    "references",
    "reference",
    "bibliography",
    "acknowledgements",
    "acknowledgments",
}


def get_paper_id(path: Path) -> str:
    """Derive a paper_id from filename, e.g. 2309.02144v1.grobid.json -> 2309.02144v1"""
    name = path.name
    if name.endswith(".json"):
        name = name[:-5]
    if ".grobid" in name:
        name = name.split(".grobid")[0]
    return name


def normalize_section_name(section_name: str) -> str:
    return re.sub(r"\s+", " ", str(section_name)).strip().lower()


def extract_year(doc: dict) -> Optional[int]:
    """
    Try to extract a publication year from common fields.
    If not found, return None.
    """
    candidates = []

    # Direct common keys
    for k in ["year", "publication_year", "published_year", "pub_year"]:
        if k in doc:
            candidates.append(doc.get(k))

    # Nested metadata fields
    meta = doc.get("metadata") or doc.get("meta") or {}
    if isinstance(meta, dict):
        for k in ["year", "publication_year", "published_year", "pub_year"]:
            if k in meta:
                candidates.append(meta.get(k))

    # Try to convert any candidate to int safely
    for c in candidates:
        if c is None:
            continue
        try:
            y = int(str(c).strip())
            if 1800 <= y <= 2100:
                return y
        except Exception:
            continue

    return None


def iterate_sentences(doc: dict) -> Iterable[Tuple[str, str, List[str]]]:
    """
    Yield (section_name, sentence_text, tokens) from the JSON structure.

    This is intentionally tolerant because formats differ across pipelines.
    We only consider entries where a section maps to a list of sentence-like items.
    """
    # Prefer doc["sections"] if it exists and is a dict
    if isinstance(doc, dict) and isinstance(doc.get("sections"), dict):
        sections = doc["sections"]
    elif isinstance(doc, dict):
        # Fall back: some pipelines store sections at the top-level
        sections = doc
    else:
        return

    for section_name, section_content in sections.items():
        # We only treat list content as "sentences"
        if not isinstance(section_content, list):
            continue

        for sent in section_content:
            # Case 1: sentence is a dict with "tokens" and maybe "text"
            if isinstance(sent, dict):
                tokens = sent.get("tokens")
                sent_text = sent.get("text")

                if tokens is None and sent_text is not None:
                    tokens = str(sent_text).split()

                if sent_text is None and tokens is not None:
                    sent_text = " ".join([str(t) for t in tokens])

                if tokens is not None:
                    tokens = [str(t) for t in tokens]

            # Case 2: sentence is a list of token strings
            elif isinstance(sent, list):
                tokens = [str(t) for t in sent]
                sent_text = " ".join(tokens)

            # Case 3: sentence is just a string
            else:
                sent_text = str(sent)
                tokens = sent_text.split()

            if not tokens:
                continue

            yield section_name, str(sent_text), tokens


def get_section_params(section_name_lower: str) -> Tuple[int, int]:
    """
    Return (max_tokens, overlap_tokens) for this section.
    """
    if section_name_lower == "abstract":
        return MAX_TOKENS_ABSTRACT, OVERLAP_ABSTRACT
    return MAX_TOKENS_BODY, OVERLAP_BODY


def chunk_section(
    sent_items: List[Tuple[str, List[str]]],
    max_tokens: int,
    overlap_tokens: int
) -> Iterable[str]:
    """
    Sentence-based chunking with sentence-boundary overlap.

    sent_items: list of (sentence_text, tokens)
    """
    buf: List[Tuple[str, List[str]]] = []
    tok_count = 0

    def flush_current():
        nonlocal buf, tok_count
        if not buf:
            return None
        text = " ".join(s for s, _ in buf).strip()
        return text

    def build_overlap_buffer():
        nonlocal buf, tok_count
        if overlap_tokens <= 0 or not buf:
            buf = []
            tok_count = 0
            return

        overlap_buf: List[Tuple[str, List[str]]] = []
        t = 0
        # Take whole sentences from the end until we reach overlap_tokens
        for s, toks in reversed(buf):
            if overlap_buf and (t + len(toks) > overlap_tokens):
                break
            overlap_buf.append((s, toks))
            t += len(toks)
            if t >= overlap_tokens:
                break

        overlap_buf.reverse()
        buf = overlap_buf
        tok_count = sum(len(toks) for _, toks in buf)

    for sent_text, tokens in sent_items:
        # If adding this sentence would exceed chunk size flush current chunk
        if buf and tok_count + len(tokens) > max_tokens:
            out_text = flush_current()
            if out_text:
                yield out_text
            build_overlap_buffer()

        buf.append((sent_text, tokens))
        tok_count += len(tokens)

    out_text = flush_current()
    if out_text:
        yield out_text


def safe_section_for_id(section_name: str) -> str:
    """
    Keep chunk_id stable and easy to parse:
    - lowercase
    - spaces -> underscore
    - remove non [a-z0-9_]
    """
    s = normalize_section_name(section_name)
    s = s.replace(" ", "_")
    s = re.sub(r"[^a-z0-9_]+", "", s)
    return s or "section"


def main():
    if not IN_DIR.exists():
        raise FileNotFoundError(f"Input folder not found: {IN_DIR}")

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)

    with OUT_PATH.open("w", encoding="utf-8") as out_f:
        for json_file in sorted(IN_DIR.glob("*.json")):
            paper_id = get_paper_id(json_file)
            print(f"Processing {json_file.name} (paper_id={paper_id})")

            with json_file.open("r", encoding="utf-8") as f:
                doc = json.load(f)

            year = extract_year(doc)

            # Collect sentences grouped by section in first-seen order
            section_order: List[str] = []
            grouped: Dict[str, List[Tuple[str, List[str]]]] = OrderedDict()

            any_sentence = False
            for section_name, sent_text, tokens in iterate_sentences(doc):
                any_sentence = True
                if section_name not in grouped:
                    grouped[section_name] = []
                    section_order.append(section_name)
                grouped[section_name].append((sent_text, tokens))

            if not any_sentence:
                print(f"  WARNING: no sentences found in {json_file.name}")
                continue

            chunk_index_global = 0
            wrote = 0

            for section_name in section_order:
                section_lower = normalize_section_name(section_name)

                # Skip unhelpful sections
                if section_lower in SKIP_SECTIONS:
                    continue

                max_tokens, overlap = get_section_params(section_lower)

                # Make chunks for this section only
                for chunk_text in chunk_section(grouped[section_name], max_tokens, overlap):
                    section_id = safe_section_for_id(section_name)
                    chunk_id = f"{paper_id}_{section_id}_{chunk_index_global:04d}"

                    record = {
                        "chunk_id": chunk_id,
                        "paper_id": paper_id,
                        "section": section_id,      # normalized section label
                        "section_raw": section_name,  # original section key from JSON
                        "text": chunk_text,
                    }

                    if year is not None:
                        record["year"] = year

                    out_f.write(json.dumps(record, ensure_ascii=False) + "\n")
                    chunk_index_global += 1
                    wrote += 1

            print(f"  -> wrote {wrote} chunks")

    print(f"\nAll done. Chunks written to {OUT_PATH}")


if __name__ == "__main__":
    main()