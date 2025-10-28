#!/usr/bin/env python3
"""
parse_pdfs.py
~~~~~~~~~~~~~~

This script iterates over a directory of raw PDF files, extracts the
plain text using the ``pdfminer.six`` library, cleans the text, and
attempts to identify basic structural elements (title, abstract,
keywords, introduction, methods, results, discussion, conclusion, and
references).  The structured representation is written out to a JSON
file per paper in a target directory.  This script is intended to
serve the Text Extraction portion of the project.

The structure extraction is heuristic and relies on regular
expressions to match common section headings.  For more robust
parsing of scientific articles, consider integrating a tool such as
GROBID.

Usage example:

    python parse_pdfs.py --in_dir data/raw/pdfs --out_dir data/parsed

Requirements:
    - pdfminer.six (`pip install pdfminer.six`)

"""

import argparse
import json
import os
import re
from typing import Dict, Tuple

try:
    from pdfminer.high_level import extract_text  # type: ignore
except ImportError:
    raise SystemExit(
        "The 'pdfminer.six' package is required. Install it with 'pip install pdfminer.six'."
    )


def clean_text(text: str) -> str:
    """Normalize whitespace and remove some PDF artefacts."""
    # Replace multiple spaces and newlines with a single space
    text = re.sub(r"\s+", " ", text)
    # Remove form feed characters
    text = text.replace("\f", " ")
    # Remove stray hyphenation at line breaks (word-\n continuation)
    text = re.sub(r"(\w+)-\s*\n(\w+)", r"\1\2", text)
    return text.strip()


def extract_structure(text: str) -> Dict[str, str]:
    """Attempt to extract common sections from the full text."""
    # Define regular expression patterns for common sections in academic papers
    section_patterns = [
        # Title (assume everything up to the abstract)
        r"^(.*?)(?=Abstract|ABSTRACT|abstract)",
        # Abstract
        r"(?:Abstract|ABSTRACT)[:\s]*(.*?)(?=(?:Introduction|INTRODUCTION|Keywords|KEYWORDS|\d+\.|\n\n\d+\s+\w+))",
        # Keywords
        r"(?:Keywords|KEYWORDS)[:\s]*(.*?)(?=(?:Introduction|INTRODUCTION|\d+\.|\n\n\d+\s+\w+))",
        # Introduction
        r"(?:\n|^)(?:\d+\s+)?(?:Introduction|INTRODUCTION)[\s:.]*(.*?)(?=(?:\n\d+\.|\n\d+\s+\w+|\n\n\d+\s+\w+))",
        # Methods/Methodology
        r"(?:\n|^)(?:\d+\s+)?(?:Methods|Method|Methodology|METHODS|METHOD)[\s:.]*(.*?)(?=(?:\n\d+\.|\n\d+\s+\w+|\n\n\d+\s+\w+))",
        # Results
        r"(?:\n|^)(?:\d+\s+)?(?:Results|RESULTS)[\s:.]*(.*?)(?=(?:\n\d+\.|\n\d+\s+\w+|\n\n\d+\s+\w+))",
        # Discussion
        r"(?:\n|^)(?:\d+\s+)?(?:Discussion|DISCUSSION)[\s:.]*(.*?)(?=(?:\n\d+\.|\n\d+\s+\w+|\n\n\d+\s+\w+))",
        # Conclusion
        r"(?:\n|^)(?:\d+\s+)?(?:Conclusion|Conclusions|CONCLUSION|CONCLUSIONS)[\s:.]*(.*?)(?=(?:\n\d+\.|\n\d+\s+\w+|\n\n\d+\s+\w+|References|REFERENCES|Bibliography|BIBLIOGRAPHY))",
        # References
        r"(?:\n|^)(?:\d+\s+)?(?:References|REFERENCES|Bibliography|BIBLIOGRAPHY)[\s:.]*(.*?)(?=$)",
    ]

    section_names = [
        "title",
        "abstract",
        "keywords",
        "introduction",
        "methods",
        "results",
        "discussion",
        "conclusion",
        "references",
    ]

    structure = {name: "" for name in section_names}
    for i, pattern in enumerate(section_patterns):
        match = re.search(pattern, text, re.DOTALL)
        if match:
            content = match.group(1).strip()
            structure[section_names[i]] = clean_text(content)

    return structure


def process_pdf(pdf_path: str) -> Dict[str, str]:
    """Extract and structure the contents of a single PDF file."""
    try:
        raw_text = extract_text(pdf_path)
    except Exception as exc:
        print(f"Error extracting text from {pdf_path}: {exc}")
        raw_text = ""
    raw_text = clean_text(raw_text)
    structure = extract_structure(raw_text)
    structure["full_text"] = raw_text
    structure["source_pdf"] = pdf_path
    # Use the file stem as an identifier
    structure["filename"] = os.path.splitext(os.path.basename(pdf_path))[0]
    return structure


def main() -> None:
    parser = argparse.ArgumentParser(description="Parse PDFs into structured JSON.")
    parser.add_argument(
        "--in_dir",
        required=True,
        help="Directory containing PDF files (e.g. data/raw/pdfs)",
    )
    parser.add_argument(
        "--out_dir",
        required=True,
        help="Directory to write parsed JSON files (e.g. data/parsed)",
    )
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    pdf_files = [f for f in os.listdir(args.in_dir) if f.lower().endswith(".pdf")]
    if not pdf_files:
        print(f"No PDF files found in {args.in_dir}")
        return

    for pdf_name in pdf_files:
        pdf_path = os.path.join(args.in_dir, pdf_name)
        structure = process_pdf(pdf_path)
        out_path = os.path.join(args.out_dir, f"{structure['filename']}.json")
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(structure, f, indent=2, ensure_ascii=False)
        print(f"Parsed {pdf_name} → {os.path.relpath(out_path, args.out_dir)}")


if __name__ == "__main__":
    main()