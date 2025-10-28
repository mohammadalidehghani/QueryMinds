#!/usr/bin/env python3
"""
download_papers.py
~~~~~~~~~~~~~~~~~~~

This script provides a simple command‑line interface for downloading
academic papers from the arXiv API and saving both the PDFs and their
metadata.  It is designed to support the Data Acquisition & Repository
Setup tasks for a RAG system.

Usage example:

    # Download the first 100 papers matching the query "retrieval augmented generation"
    # and save them into the data/raw directory
    python download_papers.py --query "retrieval augmented generation" \
        --max_results 100 --out_dir data/raw
    #python download_papers.py --query "machine learning" --max_results 300 --out_dir data/raw


The script will create a ``pdfs`` subdirectory under the output directory
and write a JSONL metadata file named ``arxiv_meta.jsonl``.

Requirements:
    - The ``arxiv`` Python package must be installed.  Install it via:
      ``pip install arxiv``

Notes:
    - This script performs polite rate limiting with a 1‑second delay
      between PDF downloads to avoid overloading the arXiv servers.
    - If a PDF already exists locally it will not be downloaded again.

"""

import argparse
import json
import os
import time
from typing import List, Optional

try:
    import arxiv  # type: ignore
except ImportError as e:
    raise SystemExit(
        "The 'arxiv' package is required for this script. Install it with 'pip install arxiv'."
    )


class ArxivAPI:
    """Simple wrapper around the arXiv API for searching and downloading papers."""

    def __init__(self, save_dir: str) -> None:
        self.save_dir = os.path.abspath(save_dir)
        os.makedirs(self.save_dir, exist_ok=True)

    import arxiv

    def search_papers(self, query, max_results=10, categories=None):
        if categories:
            cat_query = " OR ".join(f"cat:{c}" for c in categories)
            full_query = f"{query} AND ({cat_query})"
        else:
            full_query = query

        # Build a Search object without max_results
        search = arxiv.Search(
            query=full_query,
            max_results=None,
            sort_by=arxiv.SortCriterion.Relevance,
            sort_order=arxiv.SortOrder.Descending,
        )

        # Use the Client for robust pagination
        client = arxiv.Client(page_size=100, delay_seconds=3, num_retries=3)

        results = []
        try:
            for paper in client.results(search):
                results.append(paper)
                if len(results) >= max_results:
                    break
        except Exception as e:
            print(f"Warning: error while fetching results ({e}); returning partial list.")
        return results

    def download_paper(self, paper: arxiv.Result, save_path: Optional[str] = None) -> Optional[str]:
        """Download a single paper's PDF.

        Returns the local path to the downloaded file, or ``None`` on failure.
        """
        if save_path is None:
            save_path = os.path.join(self.save_dir, f"{paper.get_short_id()}.pdf")

        if os.path.exists(save_path):
            # Already downloaded
            return save_path

        try:
            paper.download_pdf(filename=save_path)
            # Polite rate limiting
            time.sleep(1.0)
            return save_path
        except Exception as exc:
            print(f"Error downloading {paper.get_short_id()}: {exc}")
            return None

    def get_paper_metadata(self, paper: arxiv.Result) -> dict:
        """Extract a dictionary of useful metadata from an arXiv result."""
        return {
            "id": paper.get_short_id(),
            "title": paper.title,
            "authors": [a.name for a in paper.authors],
            "abstract": paper.summary.strip(),
            "categories": paper.categories,
            "published": paper.published.strftime("%Y-%m-%d"),
            "pdf_url": paper.pdf_url,
            "entry_id": paper.entry_id,
        }


def main() -> None:
    parser = argparse.ArgumentParser(description="Download academic papers from arXiv.")
    parser.add_argument("--query", required=True, help="Search query string")
    parser.add_argument(
        "--max_results",
        type=int,
        default=100,
        help="Maximum number of papers to download",
    )
    parser.add_argument(
        "--categories",
        nargs="*",
        help="Optional list of arXiv categories to filter by (e.g. cs.AI cs.CL)",
    )
    parser.add_argument(
        "--out_dir",
        default=os.path.join("data", "raw"),
        help="Output directory to store PDFs and metadata (default: data/raw)",
    )

    args = parser.parse_args()

    # Ensure the output directory exists
    os.makedirs(args.out_dir, exist_ok=True)
    pdf_dir = os.path.join(args.out_dir, "pdfs")
    os.makedirs(pdf_dir, exist_ok=True)

    api = ArxivAPI(pdf_dir)
    print(f"Searching arXiv for query: '{args.query}'")
    papers = api.search_papers(args.query, args.max_results, args.categories)

    if not papers:
        print("No papers found for the given query.")
        return

    print(f"Found {len(papers)} papers. Starting download...")
    all_metadata = []
    for i, paper in enumerate(papers, start=1):
        local_pdf = api.download_paper(paper)
        meta = api.get_paper_metadata(paper)
        if local_pdf:
            meta["local_pdf"] = local_pdf
        all_metadata.append(meta)
        print(f"Downloaded {i}/{len(papers)}: {paper.get_short_id()}")

    # Write metadata as JSONL
    meta_path = os.path.join(args.out_dir, "arxiv_meta.jsonl")
    with open(meta_path, "w", encoding="utf-8") as f:
        for meta in all_metadata:
            f.write(json.dumps(meta, ensure_ascii=False) + "\n")
    print(f"Saved metadata to {meta_path}")


if __name__ == "__main__":
    main()