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

#  Final Overview

Final version of the projects builds upon the first two milestones, using the existing embedding retrieval for implementing two RAG (Retrieval Augmented Generation) systems:
- **Vanilla RAG**, allowing free-form generation
- **Strict Constrained RAG**, restricting generation to retrieved evidence

The two systems were implemented in three different versions:
- ***Version 1***: Both systems implemented on the first versions of questions, and one manual labeling
- ***Version 2***: Both systems implemented on a refined set of questions, and a trained embedding retrieval model with three large language models labeling the golden truth labels
- ***Version 3***: Both systems implemented on a manually labeled trained embedding retrieval model
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
