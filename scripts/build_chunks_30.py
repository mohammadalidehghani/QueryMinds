import json
from pathlib import Path

# Project root
BASE_DIR = Path(__file__).resolve().parent.parent

# Our 30 selected JSON files
IN_DIR = BASE_DIR / "data" / "selected_30"

# Where we save all chunks (one big JSONL file)
OUT_PATH = BASE_DIR / "data" / "chunks_30.jsonl"

# Chunk settings – you can tune these
MAX_TOKENS = 200   # target chunk length
OVERLAP = 50       # how many tokens overlap between chunks

def get_paper_id(path: Path) -> str:
    """Derive a paper_id from filename, e.g. 2309.02144v1.grobid.json -> 2309.02144v1"""
    name = path.name
    if name.endswith(".json"):
        name = name[:-5]
    if ".grobid" in name:
        name = name.split(".grobid")[0]
    return name

def iterate_sentences(doc):
    """
    Go through the JSON structure and yield (section_name, sentence_text, tokens).

    This function tries to be flexible, because we don't know the exact format 100%.
    """
    # Many pipelines store content in doc["sections"], but if not, we just use the doc itself
    sections = doc.get("sections", doc)

    # sections is expected to be a dict: section_name -> list_of_sentences
    for section_name, section_content in sections.items():
        # section_content should be a list of sentence objects
        if not isinstance(section_content, list):
            continue

        for sent in section_content:
            # Case 1: sentence is a dict with "tokens" and maybe "text"
            if isinstance(sent, dict):
                tokens = sent.get("tokens")
                sent_text = sent.get("text")
                if tokens is None and sent_text is not None:
                    tokens = sent_text.split()
                if sent_text is None and tokens is not None:
                    sent_text = " ".join(tokens)
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

            yield section_name, sent_text, tokens

def make_chunks(sentences):
    """
    Given an iterator of (section_name, sent_text, tokens),
    group them into chunks with MAX_TOKENS and OVERLAP.
    """
    current_tokens = []
    current_texts = []
    current_section = None

    for section_name, sent_text, tokens in sentences:
        # Start new section if needed
        if current_section is None:
            current_section = section_name

        # If section changes and we already have content → flush current chunk
        if section_name != current_section and current_tokens:
            yield current_section, " ".join(current_texts)
            current_tokens = []
            current_texts = []
            current_section = section_name

        # If adding this sentence would exceed chunk size → flush current chunk
        if current_tokens and len(current_tokens) + len(tokens) > MAX_TOKENS:
            # Emit chunk
            yield current_section, " ".join(current_texts)

            # Keep overlap tokens from the end
            if OVERLAP > 0:
                overlap_tokens = current_tokens[-OVERLAP:]
                overlap_text = " ".join(overlap_tokens)
                current_tokens = overlap_tokens[:]
                current_texts = [overlap_text]
            else:
                current_tokens = []
                current_texts = []

        # Add sentence to current chunk
        current_tokens.extend(tokens)
        current_texts.append(sent_text)

    # Flush last chunk
    if current_tokens:
        yield current_section, " ".join(current_texts)

def main():
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)

    with OUT_PATH.open("w", encoding="utf-8") as out_f:
        for json_file in sorted(IN_DIR.glob("*.json")):
            paper_id = get_paper_id(json_file)
            print(f"Processing {json_file.name} (paper_id={paper_id})")

            with json_file.open("r", encoding="utf-8") as f:
                doc = json.load(f)

            sentences = list(iterate_sentences(doc))
            if not sentences:
                print(f"  WARNING: no sentences found in {json_file.name}")
                continue

            chunk_index = 0
            for section_name, chunk_text in make_chunks(sentences):
                chunk_id = f"{paper_id}_{section_name}_{chunk_index:04d}"

                record = {
                    "chunk_id": chunk_id,
                    "paper_id": paper_id,
                    "section": section_name,
                    "text": chunk_text,
                }

                out_f.write(json.dumps(record, ensure_ascii=False) + "\n")
                chunk_index += 1

            print(f"  -> wrote {chunk_index} chunks")

    print(f"\nAll done. Chunks written to {OUT_PATH}")

if __name__ == "__main__":
    main()
