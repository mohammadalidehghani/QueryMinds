# QueryMinds — Milestone 1: Academic PDFs to CoNLL-U Corpus

This milestone establishes a complete NLP preprocessing pipeline that transforms **raw academic PDF documents** into linguistically annotated **CoNLL-U files**.  
The project implements automated text extraction, cleaning, sentence segmentation, tokenization, normalization, lemmatization, and statistical analysis.

---

## Overview

The developed pipeline enables:
- Extraction of structured text from scientific PDFs.  
- Tokenization and sentence segmentation using **NLTK**.  
- Lemmatization and POS-tagging using **Stanza**.  
- Conversion into **CoNLL-U** format compliant with Universal Dependencies standards.  
- Corpus validation, statistical reporting, and visualization.

Final output:  
A linguistically normalized and validated corpus of **191 CoNLL-U documents**, ready for downstream NLP tasks such as semantic search, clustering, and retrieval-augmented generation (RAG).

---

## Environment Setup

```bash
# Create and activate a Python environment
conda create -n rag_env python=3.10 -y
conda activate rag_env

# Install all dependencies
pip install -r requirements.txt

# Download NLTK Punkt tokenizer
python - <<PY
import nltk, ssl
try:
    ssl._create_default_https_context = ssl._create_unverified_context
except:
    pass
nltk.download('punkt')
PY
