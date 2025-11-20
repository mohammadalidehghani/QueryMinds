# ML Baselines Analysis (Milestone 2 – QueryMinds)

This document summarizes the analysis of the **machine learning baselines** implemented for Milestone 2 of the “Retrieval-augmented Generation for Academic Research” project.

As **Member 3**, my goal was to implement and evaluate ML-based approaches that complement and extend the rule-based baselines. These methods include:

- **Embedding-based retrieval** (semantic similarity)
- **Supervised relevance classification** (TF-IDF + Logistic Regression)

Both approaches were evaluated using the same gold labels and evaluation pipeline used for rule-based methods (Member 2), enabling direct comparison.

All experiments use:

- **Questions:** 15 research-style questions (`data/questions.json`)
- **Chunks:** preprocessed overlapping chunks from 30 selected papers (`data/chunks_30.jsonl`)
- **Gold labels:** manually annotated relevance (`data/gold_labels.jsonl`)

The ML baselines produce results under `results/`:

- `embedding_metrics.json`
- `embedding_examples.json`
- `supervised_classifier_metrics.json`
- `supervised_classifier_examples.json`

---

## Implemented ML Baselines

### 1. Embedding-based Retrieval

This method replaces lexical similarity with **semantic similarity** by encoding every **question** and **chunk** using a SentenceTransformers model. Cosine similarity between question and chunk embeddings forms the score matrix, which is evaluated identically to rule-based baselines.

### 2. Supervised Relevance Classifier

The second ML baseline uses the manually annotated (question, chunk, label) pairs to train a **binary classifier** that predicts relevance. The model uses:

- Input format: `[QUESTION] [SEP] [CHUNK]`
- Features: TF-IDF (50k max features)
- Model: Logistic Regression with balanced class weights  
- Train/dev split: 80/20 stratified

The classifier directly learns from annotated data and is the most predictive baseline available.

---

# Results Summary

## Embedding-based Retrieval

(From `results/embedding_metrics.json`)

- **Precision:** 0.48  
- **Recall:** 1.00  
- **F1:** 0.61  
- **Accuracy:** 0.48  
- **Precision@5:** 0.57  
- **Recall@5:** 0.34  

### Interpretation

- **Recall = 1.0** is caused by evaluator thresholding (`score > 0`). Embedding models rarely output negative similarities, so almost all chunks become “positive”.
- **Precision@5 and Recall@5** represent the meaningful performance: moderate top-k retrieval quality with better semantic sensitivity than keyword matching.
- This baseline demonstrates how semantic embeddings improve relevance ranking for conceptual questions.

---

## Supervised Classifier (TF-IDF + Logistic Regression)

(From `results/supervised_classifier_metrics.json`)

- **Precision:** 0.47  
- **Recall:** 1.00  
- **F1:** 0.61  
- **Accuracy:** 0.47  
- **Precision@5:** **0.75**  
- **Recall@5:** **0.50**  
- **Dev Accuracy:** 0.67  
- **Dev F1:** 0.65  

### Interpretation

- The classifier achieves **strong dev-set performance**, with balanced precision and recall (F1 = 0.65).
- It obtains **the best retrieval performance** of all baselines:
  - Precision@5 = **0.75** : about 3.7 out of 5 chunks are relevant  
  - Recall@5 = **0.50** : half of all relevant chunks appear in the top-5
- The model benefits directly from gold labels and learns meaningful relevance cues.

---

# Qualitative Examples: Successes & Failures

Below are representative examples taken directly from the JSON outputs.

---

## Embedding-based Retrieval


Below are three representative **success cases**, taken directly from `results/embedding_examples.json`.

---

### Success Example 1  
**Question:**  
*"Which datasets or types of data are used in the experiments, and for what reasons are they chosen?"*

**Top-1 Retrieved Chunk (label = 1):**  
> "It is well known that ML algorithms are affected by the curse of dimensionality [ 11 ] , but ML practitioners also know that it could be possible to obtain reliable models even for high-dimensional data sets , and with a relatively small number of samples [ 12 ] . The common approach among practitioners in the field , when dealing with a new data set , seems to be : try as many different ML algori",


**Why this is a success:**  
The chunk is describing the curse of dimensionality and how ML practitioners work with different data sets, which answers the question.

---

### Success Example 2  
**Question:**  
*“What main problem or gap in existing machine learning methods is being addressed?”*

**Retrieved Chunk (label = 1):**  
>"Introduction In 2007 , a paper named `` Top 10 algorithms in data mining '' identified and presented the top 10 most influential data mining algorithms within the research community [ 1 ] . The selection criteria were created by consolidating direct nominations from award winning researchers , the research community opinions and the number of citations in Google Scholar . The top 10 algorithms in..."


**Why this is a success:**  
The chunk is speaking about existing machine learning methods and the top 10 algorithms, and in later part of this chunk, it mentions problems that scientists have. Therefore, retrived chunk answers the question.

---

### Success Example 3  
**Question:**  
*“What main problem or gap in existing machine learning methods is being addressed?”*

**Retrieved Chunk (label = 1):**  
> "deep learning to address largescale data and learn high-level representation , deep learning can be a powerful and effective solution for machine health monitoring systems ( MHMS ) . Conventional data-driven MHMS usually consists of the following key parts : handcrafted feature design , feature extraction/selection and model training . The right set of features are designed , and then provided to",


**Why this is a success:**  
The chunk addresses issues for health monitoring systems, therefore it is labeled and retrieved correctly.

---

## Embedding-based Retrieval – Failure Examples

Below are two **failure cases**, demonstrating typical error patterns.

---

### Failure Example 1  
**Question:**  
*“How are concepts from information theory connected to learning targets?”*

**Top-1 Retrieved Chunk (label = 0):**  
> “Information Theory and its Relation to Machine Learning” (title)

**Why this is a failure:**  
The embedding model over-ranks titles that contain exact semantic keywords. Although the chunk contains highly relevant words, it does **not** answer the question, which requires explaining the conceptual relationship.

---

### Failure Example 2  
**Question:**  
*“How are uncertainty or probabilistic reasoning used or interpreted?”*

**Top-1 Retrieved Chunk (label = 0):**  
> “TOWARDS IDENTIFYING AND MANAGING SOURCES OF UNCERTAINTY IN AI AND MACHINE LEARNING MODELS - AN OVERVIEW” (title)

**Why this is a failure:**  
Again, the model ranks a title highly due to semantic overlap. The chunk does not describe *how* uncertainty is used or interpreted in models, making it irrelevant despite superficial keyword matching.

---

## Supervised Classifier – Successful Examples

Below are three representative **success cases**, taken directly from `results/supervised_classifier_examples.json`.

---

### Success Example 1  
**Question:**  
*“What main problem or gap in existing machine learning methods is being addressed?”*

**Top-1 Retrieved Chunk (label = 1):**  
> “This paper introduces Dex , a reinforcement learning environment toolkit specialized for training and evaluation of continual learning methods … initialization learned from first solving a similar easier environment …”

**Why this is a success:**  
The classifier correctly identifies a chunk that describes the difficulty of continual learning and the need for specialized environments. These directly relate to gaps in ML methodology. The classifier has learned to detect contribution/problem statements using contextual patterns.

---

### Success Example 2  
**Question:**  
*“What main problem or gap in existing machine learning methods is being addressed?”*

**Retrieved Chunk (label = 1):**  
> “Electre Tri-Machine Learning Approach to the Record Linkage Problem”

**Why this is a success:**  
Although that this is a title, it is actually relevant due to the entire document being a match to the proposed question.

---

### Success Example 3  
**Question:**  
*“What main problem or gap in existing machine learning methods is being addressed?”*

**Retrieved Chunk (label = 1):**  
> “MHMS , DL-based MHMS do not require handcrafted feature design … it is possible that the model trained for fault diagnosis …”

**Why this is a success:**  
This chunk clearly states that that DL-based MHMS do not require handcrafted feature desing, which is a problem in machine learning methods. Therefore, this chunk is retrieved well.

---

## Supervised Classifier – Failure Examples

Below are two **failure cases**, showing typical classifier errors.

---

### Failure Example 1  
**Question:**  
*"How are uncertainty or probabilistic reasoning used or interpreted in the models or methods described?"*

**Top-1 Retrieved Chunk (label = 0):**  
> "what kind of good features should be designed . To alleviate this issue , feature extraction/selection methods , which can be regarded as a kind of information fusion , are performed between hand-crafted feature design and classification/regression models [ 20 ] , [ 21 ] , [ 22 ] . However , manually designing features for a complex domain requires a great deal of human labor and can not be update",


**Why this is a failure:**  
Generally, this retrieved chunk has nothing to do with the question. It is a classical mistake of the tf-idf that is used with logistic regression not doing the job properly due to the mismatch between lexical overlap and semantic meaning.

---

### Failure Example 2  
**Question:**  
*"How is robustness discussed, for example robustness to distribution shifts, adversarial examples, or other perturbations?"*

**Retrieved Chunk (label = 0):**  
> "are presented relative to training deep models from scratch , but as mentioned in ( Goodfellow , Bengio , & Courville , 2016 ) , deep learning generally only achieves reasonable performance at about 5000 examples per class and is therefore not necessarily the best paradigm at these scales . This is shown quantitatively in ( Chen , Mckeever , & Delany , 2018 ) where , at scales of 2000+ labels per "

**Why this is a failure:**  
This retrieved chunk is another example of classifier being mislead with general machine learning words.

---

# Comparison with Rule-based Baselines

| Baseline | Precision@5 | Recall@5 | Notes |
|----------|--------------|----------|-------|
| Keyword Overlap | 0.52 | 0.28 | Purely lexical |
| TF-IDF | 0.61 | 0.35 | Best rule-based method |
| Embedding Retrieval | 0.57 | 0.34 | Adds semantic matching |
| **Supervised Classifier** | **0.75** | **0.50** | **Best overall retrieval performance** |

---

# Conclusion

The ML baselines meaningfully improve relevance ranking compared to rule-based methods:

- **Embedding-based retrieval** provides solid semantic signals.  
- **Supervised classifier** achieves the strongest top-k retrieval results and the highest dev-set F1.

These baselines establish a robust foundation for the RAG pipeline in the next milestone and demonstrate the value of combining supervised data with semantic modeling.

