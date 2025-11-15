"""
Rule-based baselines for Milestone 2 (Member 2).

- Loads questions, chunks and gold labels
- Keyword overlap baseline (scores matrix)
- TF-IDF + cosine similarity baseline (scores matrix)
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
DATA_DIR = PROJECT_ROOT / "data"


# ---------------------------------------------------------------------------
# Helpers for loading raw data
# ---------------------------------------------------------------------------

def _load_questions(path: Path) -> List[dict]:
    """Load questions.json as a list of dicts."""
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _load_chunks(path: Path) -> List[dict]:
    """Load chunks_30.jsonl as a list of dicts."""
    chunks: List[dict] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            chunks.append(obj)
    return chunks


def _load_gold_labels(path: Path) -> List[dict]:
    """Load gold_labels.jsonl as a list of dicts."""
    labels: List[dict] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            labels.append(obj)
    return labels


# ---------------------------------------------------------------------------
# API expected by eval_rule_baselines.py
# ---------------------------------------------------------------------------

def load_data(data_dir: Path = DATA_DIR):
    """
    Load questions, chunks and gold labels from data_dir and return:

        clabels   : List[(question_id, chunk_id, label)]
        qtexts    : List[str]          (question texts, aligned with q_ids)
        ctexts    : List[str]          (chunk texts, aligned with chunk_ids)
        q_ids     : List[str]          (question ids, same order as qtexts)
        chunk_ids : List[str]          (chunk ids, same order as ctexts)
    """
    questions_path = data_dir / "questions.json"
    chunks_path = data_dir / "chunks_30.jsonl"
    labels_path = data_dir / "gold_labels.jsonl"

    questions = _load_questions(questions_path)
    chunks = _load_chunks(chunks_path)
    labels = _load_gold_labels(labels_path)

    # Order of questions & chunks is the order in the files
    q_ids = [q["id"] for q in questions]
    qtexts = [q["question"] for q in questions]

    chunk_ids = [c["chunk_id"] for c in chunks]
    ctexts = [c["text"] for c in chunks]

    # (question_id, chunk_id, label)
    clabels: List[Tuple[str, str, int]] = []
    for obj in labels:
        qid = obj["question_id"]
        cid = obj["chunk_id"]
        lab = int(obj["label"])
        clabels.append((qid, cid, lab))

    return clabels, qtexts, ctexts, q_ids, chunk_ids


# ---------------------------------------------------------------------------
# Keyword overlap baseline
# ---------------------------------------------------------------------------

def keyword_overlap_score(keywords: List[str], text: str) -> int:
    """
    Simple keyword overlap: count how many keywords from the list
    appear as whole words in the text (case-insensitive).
    """
    if not keywords:
        return 0

    text_l = text.lower()
    score = 0
    for kw in keywords:
        kw_l = kw.lower().strip()
        if not kw_l:
            continue
        pattern = r"\b" + re.escape(kw_l) + r"\b"
        if re.search(pattern, text_l):
            score += 1
    return score


def _load_keywords_by_question_text(data_dir: Path) -> Dict[str, List[str]]:
    """
    Create a mapping from question text -> keywords (if available in questions.json).
    If 'keywords' field is missing, we'll fall back to splitting the question text.
    """
    questions_path = data_dir / "questions.json"
    questions = _load_questions(questions_path)

    mapping: Dict[str, List[str]] = {}
    for q in questions:
        qtext = q["question"]
        kws = q.get("keywords")
        if isinstance(kws, list) and kws:
            mapping[qtext] = kws
        else:
            # Fallback: use simple tokenization of question text as keywords
            tokens = [t for t in re.findall(r"\w+", qtext) if len(t) > 2]
            mapping[qtext] = tokens
    return mapping


def keyword_overlap_scores(
    qtexts: List[str],
    ctexts: List[str],
    data_dir: Path = DATA_DIR,
) -> np.ndarray:
    """
    Compute a score matrix S with shape (num_questions, num_chunks),
    where S[i, j] = keyword overlap score between question i and chunk j.

    qtexts: list of question texts (same order as returned by load_data)
    ctexts: list of chunk texts   (same order as returned by load_data)
    """
    # Map question text -> keywords (prefer explicit 'keywords' from file)
    qtext_to_keywords = _load_keywords_by_question_text(data_dir)

    num_q = len(qtexts)
    num_c = len(ctexts)

    scores = np.zeros((num_q, num_c), dtype=float)

    for i, qtext in enumerate(qtexts):
        keywords = qtext_to_keywords.get(qtext)
        if keywords is None:
            # Should rarely happen; fallback to tokens from qtext
            keywords = [t for t in re.findall(r"\w+", qtext) if len(t) > 2]

        for j, chunk_text in enumerate(ctexts):
            s = keyword_overlap_score(keywords, chunk_text)
            scores[i, j] = float(s)

    return scores


# ---------------------------------------------------------------------------
# TF-IDF + cosine similarity baseline
# ---------------------------------------------------------------------------

def tfidf_cosine_scores(
    qtexts: List[str],
    ctexts: List[str],
) -> np.ndarray:
    """
    Compute TF-IDF vectors for all questions and chunks, then return the
    cosine similarity matrix with shape (num_questions, num_chunks).

    scores[i, j] = cosine_sim( question_i , chunk_j )
    """
    num_q = len(qtexts)
    num_c = len(ctexts)

    # Fit TF-IDF on all texts jointly
    all_texts = qtexts + ctexts
    vectorizer = TfidfVectorizer(stop_words="english")
    X = vectorizer.fit_transform(all_texts)

    Q = X[:num_q]        # questions
    C = X[num_q:]        # chunks

    # cosine_similarity(Q, C) → shape (num_q, num_c)
    scores = cosine_similarity(Q, C)
    assert scores.shape == (num_q, num_c)
    return scores
