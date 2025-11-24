# QueryMinds — Milestone 1 & 2  
## NLP Pipeline + Baseline Retrieval Models

This repository contains implementations for **Milestone 1** (PDF → CoNLL-U NLP pipeline)  
and **Milestone 2** (baseline retrieval models for RAG systems).

---
QueryMinds Team
- Mohammadali Dehghani 12432957
- Amir Saadati 12434679
- Amina Kadic 12439016
- Meliha Kasapovic 12439367
  

#  Milestone 1 — Overview

Milestone 1 develops a full NLP pipeline that transforms **raw academic PDF documents** into linguistically annotated **CoNLL-U** files.


###  Final Output
A clean, structured corpus of  
 **191 CoNLL-U files**, ready for downstream NLP tasks such as:
- semantic search  
- clustering  
- feature extraction  
- RAG dataset creation  

---
# Milestone 2 — Overview

Milestone 2 extends the Milestone 1 and introduces a complete RAG baseline framework for evaluating retrieval of relevant text segments in the papers.


### Implemented baseline models:
| Model | Description |
|-------|-------------|
| Keyword Overlap | Word-matching heuristic |
| TF–IDF Cosine | Lexical similarity |
| Embedding Similarity | MiniLM-L6-v2 embeddings |
| Supervised Classifier | TF–IDF + Logistic Regression |


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
```
## Project Pipeline 
# Milestone 1
```bash
# Download PDFs
python scripts/download_papers.py
# Extract and separate sections
python scripts/extract_sections.py
# Parse PDFs
python scripts/parse_pdfs.py
# Tokenize sentences
python scripts/sent_tok.py
# Normalize and convert to CONLL-U
python scripts/normalize_and_conllu.py
# Extract corpus statistics
python scripts/corpus_stats.py
```

# Milestone 2
```bash
# Select 30 papers for retrieval
python scripts/select_30.py
# Chunk papers 
python scripts/build_chunks_30.py
# Generate candidate chunks for each question
python scripts/find_candidates.py
# Label candidates
python scripts/label_candidates.py
# Make ML Training File
python scripts/make_training_data.py
# Run all the models (individual models can be run by replacing rag_pipeline with name of model)
python baselines/rag_pipeline.py
# Get comparison of results
python scripts/compare_baselines.py
```
