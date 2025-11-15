# Rule-based Baselines Analysis (Milestone 2 – QueryMinds)

This document summarizes the analysis of the rule-based baselines implemented for Milestone 2 of the RAG for Academic Research project.

---

## Baselines Implemented
As Member 2 (and primary contributor), the following rule-based approaches were implemented:

### 1. Keyword Overlap Retrieval
A simple lexical baseline measuring how many important words overlap between a question and each chunk.

### 2. TF-IDF + Cosine Similarity Retrieval
A vector-space baseline using TF-IDF representations of questions and chunks, followed by cosine similarity.

Both methods produce a score matrix (questions × chunks), and both are evaluated using relevance labels provided by Member 1.

---

## Results Summary
Results stored in `results/rule_based_metrics.json`:

### Keyword Overlap
- Precision: **0.54**
- Recall: **0.84**
- F1: **0.61**
- Accuracy: **0.55**
- Precision@5: **0.52**
- Recall@5: **0.28**

### TF-IDF Cosine
- Precision: **0.53**
- Recall: **0.86**
- F1: **0.60**
- Accuracy: **0.56**
- Precision@5: **0.61**
- Recall@5: **0.35**

---

## Interpretation

### Keyword Overlap Baseline
- High recall (≈0.84): finds most relevant chunks.
- Moderate precision: returns extra irrelevant chunks.
- Works well as a fully interpretable baseline.

### TF-IDF Baseline
- Better accuracy and better ranking performance (Precision@5 and Recall@5).
- Produces more relevant top-k chunks, which is crucial for RAG pipelines.
- Overall stronger retrieval baseline compared to keyword overlap.

### Conclusion
TF-IDF is the stronger rule-based retriever for this dataset, but keyword overlap provides a transparent and interpretable baseline for comparison.

All scripts, metrics, and this analysis were implemented by **Mohammadali Dehghani** of QueryMinds.

---
