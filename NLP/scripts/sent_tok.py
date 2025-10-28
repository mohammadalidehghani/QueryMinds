#!/usr/bin/env python3
"""
sent_tok.py
~~~~~~~~~~~~

This script takes the structured JSON files produced by ``parse_pdfs.py``
and performs sentence segmentation and tokenization using NLTK.  The
output is a new JSON file per document containing a list of sentences
with token lists.  This provides a granular representation suitable
for chunking and CoNLL‑U conversion in later stages.

Usage example:

    python sent_tok.py --in_dir data/parsed --out_dir data/parsed_tokens

Requirements:
    - nltk (`pip install nltk`)
    - Before running, ensure the 'punkt' tokenizer models are downloaded.
      The script attempts to download them if they are missing.

"""

import argparse
import json
import os
from typing import Dict, List

import nltk
from nltk.tokenize import sent_tokenize, word_tokenize


def ensure_nltk_data() -> None:
    """Ensure the NLTK punkt tokenizer is available; download if necessary."""
    try:
        nltk.data.find('tokenizers/punkt')
    except LookupError:
        nltk.download('punkt', quiet=True)


def process_document(doc: Dict[str, str]) -> Dict[str, List[Dict[str, List[str]]]]:
    """Given a parsed document dictionary, return sentence and token lists.

    Args:
        doc: A dictionary with section names and text fields produced by parse_pdfs.py.

    Returns:
        A dictionary mapping section names to a list of sentences, where each
        sentence entry contains the text and a list of tokens.
    """
    ensure_nltk_data()
    output: Dict[str, List[Dict[str, List[str]]]] = {}
    sections_to_process = [
        "title",
        "abstract",
        "keywords",
        "introduction",
        "methods",
        "results",
        "discussion",
        "conclusion",
    ]
    for section in sections_to_process:
        text = doc.get(section, "").strip()
        if not text:
            continue
        sentences = sent_tokenize(text)
        sent_list = []
        for s in sentences:
            tokens = word_tokenize(s)
            sent_list.append({"sentence": s, "tokens": tokens})
        output[section] = sent_list
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description="Sentence segmentation and tokenization")
    parser.add_argument(
        "--in_dir",
        required=True,
        help="Directory containing parsed JSON files (from parse_pdfs.py)",
    )
    parser.add_argument(
        "--out_dir",
        required=True,
        help="Directory to write tokenized JSON files",
    )
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    json_files = [f for f in os.listdir(args.in_dir) if f.lower().endswith(".json")]
    if not json_files:
        print(f"No parsed JSON files found in {args.in_dir}")
        return

    for json_name in json_files:
        in_path = os.path.join(args.in_dir, json_name)
        with open(in_path, "r", encoding="utf-8") as f:
            doc = json.load(f)
        tokenized = process_document(doc)
        out_path = os.path.join(args.out_dir, json_name)
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(tokenized, f, indent=2, ensure_ascii=False)
        print(f"Tokenized {json_name} → {os.path.relpath(out_path, args.out_dir)}")


if __name__ == "__main__":
    main()