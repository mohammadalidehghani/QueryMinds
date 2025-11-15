# Rule-based Baselines Analysis (Milestone 2 – QueryMinds)

This document summarizes the analysis of the rule-based baselines implemented for Milestone 2 of the “Retrieval-augmented Generation for Academic Research” project.

All experiments are based on the shared corpus and labeled data prepared in Milestone 1:

- Questions: 15 research-style questions (`data/questions.json`)
- Chunks: fixed-size overlapping chunks (`data/chunks_30.jsonl`)
- Gold labels: relevance of (question, chunk) pairs (`data/gold_labels.jsonl`)

The evaluation script is `scripts/eval_rule_baselines.py`, and metrics/results are stored under `results/`.

---

## Baselines Implemented

As **Member 2** (and primary contributor for this part), the following rule-based approaches were implemented in `scripts/rule_based.py`:

### 1. Keyword Overlap Retrieval

A simple lexical baseline that measures how many important words from a question overlap with each chunk. The score is the number of matched keywords (case-insensitive, whole-word matches). These scores are used both for ranking chunks per question and for binary relevance decisions.

### 2. TF-IDF + Cosine Similarity Retrieval

A vector-space baseline that:

- builds a TF-IDF representation over all question texts and chunk texts,
- computes cosine similarity between each question and each chunk,
- uses the similarity matrix for ranking and binary relevance decisions.

Both methods produce a score matrix of shape *(num_questions × num_chunks)* and are evaluated using the same gold labels.

---

## Results Summary

The script `scripts/eval_rule_baselines.py` computes per-question metrics and stores them in `results/rule_based_metrics.json`. For our current dataset, the metrics are:

### Keyword Overlap

- **Precision:** 0.54  
- **Recall:** 0.84  
- **F1:** 0.61  
- **Accuracy:** 0.55  
- **Precision@5:** 0.52  
- **Recall@5:** 0.28  

### TF-IDF Cosine

- **Precision:** 0.53  
- **Recall:** 0.86  
- **F1:** 0.60  
- **Accuracy:** 0.56  
- **Precision@5:** 0.61  
- **Recall@5:** 0.35  

(Top-k is always evaluated with **k = 5**, averaged over 15 questions.)

---

## Interpretation of Metrics

### Keyword Overlap Baseline

- **High recall (~0.84):** it finds most of the relevant chunks.
- **Moderate precision (~0.54):** it also retrieves many non-relevant chunks.
- **Accuracy (~0.55):** slightly above a naive baseline, which is reasonable given class imbalance.
- **Precision@5 / Recall@5:**  
  - On average, around **2.6 out of 5** retrieved chunks are relevant (Precision@5 ≈ 0.52).  
  - Only about **27%** of all relevant chunks appear in the top-5.

This baseline works well as a fully interpretable reference: it shows that the questions, chunks, and labels are consistent, and it provides a simple lexical retrieval signal.

### TF-IDF Baseline

- **Precision and recall** are on the same level as keyword overlap (slightly higher recall).
- **Accuracy** is slightly better (~0.56).
- **Precision@5 (0.61) and Recall@5 (0.35)** improve noticeably:
  - Among top-5 chunks, on average about **3 out of 5** are relevant.
  - It covers more of the truly relevant chunks within the top-5 list.

This is important for RAG, where typically only the top-k chunks are passed to the generator. TF-IDF provides a stronger rule-based retrieval baseline for our dataset.

---

## Qualitative Examples: Successes and Failures

To better understand the behavior of the baselines, we also generated qualitative examples using the updated `eval_rule_baselines.py`. These examples are stored in `results/rule_based_examples.json` and include, for each question:

- the top-k ranked chunks,
- their labels (1 = relevant, 0 = not relevant),
- and the model scores.

Below are a few **real examples** (directly taken from that JSON) illustrating typical **successes** and **failures**.

## Keyword Overlap – Successful Examples

#### Example 1 – Q1: Identifying the main problem or gap

- **Question (q1):**  
  *“What main problem or gap in existing machine learning methods is being addressed?”*

- **Top-1 chunk (correct, label = 1):**  
  `1612.07640v1_introduction_0008`  
  Preview:  
  > “what kind of good features should be designed . To alleviate this issue , feature extraction/selection methods ... However , manually designing features for a complex domain requires a great deal of human labor and can not be update …”

- **Why this is a success:**  
  The chunk explicitly discusses a **problem/gap** in existing methods (manual feature engineering, human labor), which matches the question intent. Important words such as *“features”, “issue”, “methods”* overlap with the question, giving a high keyword score (2.0) and ranking this chunk at the top.

---

#### Example 2 – Q2: Central contribution / main idea

- **Question (q2):**  
  *“What is the central contribution or main idea of the approach being described?”*

- **Top-1 chunk (correct, label = 1):**  
  `1706.05749v1_introduction_0004`  
  Preview:  
  > “as the inherent difficulty of the task . Therefore , some form of prior information must be given to the agent . This can be seen with AlphaGo [ 18 ] , where the agent never learned to play the game without first using supervised learning on human games …”

- **Why this is a success:**  
  This chunk describes the **main idea** of the proposed approach (using prior information / supervised learning before reinforcement learning). The shared vocabulary around *“learning”, “task”, “agent”* gives a high keyword-overlap score (3.0), and the chunk is correctly placed at rank 1.

---

#### Example 3 – Q5: Benchmarks and evaluation setups

- **Question (q5):**  
  *“What kinds of benchmarks or evaluation setups are used or discussed for assessing model performance?”*

- **Top-1 chunk (correct, label = 1):**  
  `1706.05749v1_abstract_0001`  
  Preview:  
  > “This paper introduces Dex , a reinforcement learning environment toolkit specialized for training and evaluation of continual learning methods as well as general reinforcement learning problems . We also present the novel continual learning method of incremental learning …”

- **Why this is a success:**  
  The chunk describes a **training and evaluation environment toolkit** (Dex), which is directly relevant to benchmarks / evaluation setups. Keyword overlap on words like *“training”, “evaluation”, “environment”* helps the rule-based method identify this as a relevant chunk.

---

### ❌ Keyword Overlap – Failure Examples

#### Example 4 – Q3: Information theory and learning targets

- **Question (q3):**  
  *“How are concepts from information theory connected to machine learning objectives or learning targets?”*

- **Top-1 chunk (incorrect, label = 0):**  
  `1501.04309v1_title_0000`  
  > “Information Theory and its Relation to Machine Learning”

- **Relevant chunks (labels = 1) appear lower:**  
  For example, the abstract and introduction of `1711.01431v1` (ranks 3–5) discuss **learning representations, model entropy, and concept formation**, which connect information-theoretic ideas to learning objectives.

- **Why this is a failure:**  
  The keyword baseline is strongly attracted by titles that contain exactly the words *“Information Theory”* and *“Machine Learning”*, but the title chunk itself was annotated as not directly answering the question. The more conceptually relevant discussion appears in other chunks that are ranked lower despite being labeled 1.

---

#### Example 5 – Q4: Uncertainty and probabilistic reasoning

- **Question (q4):**  
  *“How are uncertainty or probabilistic reasoning used or interpreted in the models or methods described?”*

- **Top-1 to Top-5 chunks (all label = 0):**  
  Top chunk: `1504.03874v1_introduction_0002`  
  > “… machine learning ( ML ) and big data mining questions are fundamental …”

  Other top chunks include titles/introductions related to belief function theory or Bayesian optimization, but are not annotated as directly answering the question.

- **Why this is a failure:**  
  Keyword overlap fires on generic terms like *“uncertainty”*, *“Bayesian”*, *“belief”*, or *“machine learning”*, but these chunks do not actually explain **how** uncertainty is used or interpreted in the models. The method overestimates relevance based on superficial matches.

---

### TF-IDF Cosine – Successful Examples

#### Example 6 – Q1: Main problem or gap (TF-IDF)

- **Question (q1):**  
  *“What main problem or gap in existing machine learning methods is being addressed?”*

- **Top-1 chunk (correct, label = 1):**  
  `1501.04309v1_abstract_0001`  
  > “In this position paper , I first describe a new perspective on machine learning ( ML ) by four basic problems ( or levels ) , namely , “What to learn?”, “How to learn?”, “What to evaluate?”, and “What to adjust?” …”

- **Why this is a success:**  
  The abstract explicitly discusses **learning targets and high-level problems/gaps** in existing machine learning formulations. TF-IDF picks up semantically central terms (*“what to learn”, “what to evaluate”*) and gives this chunk the highest similarity score among all chunks for q1.

---

#### Example 7 – Q2: Central contribution / main idea

- **Question (q2):**  
  *“What is the central contribution or main idea of the approach being described?”*

- **Top-1 chunk (correct, label = 1):**  
  `2110.12773v1_abstract_0002`  
  > “… Extending such a benchmarking approach and identifying metrics for the application of machine learning methods to scientific datasets is a new challenge … In this paper , we describe our approach to the d…”

- **Why this is a success:**  
  This abstract paragraph clearly states the **main contribution**: extending benchmarking approaches and defining metrics for scientific machine learning. TF-IDF captures the shared vocabulary around *“benchmarking”, “metrics”, “approach”* and ranks this chunk first.

---

### TF-IDF Cosine – Failure Examples

#### Example 8 – Q3: Information theory and learning targets (TF-IDF)

- **Question (q3):**  
  *“How are concepts from information theory connected to machine learning objectives or learning targets?”*

- **Top-1 chunk (incorrect, label = 0):**  
  `1501.04309v1_title_0000`  
  > “Information Theory and its Relation to Machine Learning”

- **Lower-ranked relevant chunks (label = 1):**  
  Abstract and introduction chunks of `1711.01431v1` and `1501.04309v1` describe **model entropy, concept formation, and learning targets**, which are labeled as relevant and have reasonably high similarity scores, but are still ranked below the title.

- **Why this is a failure:**  
  TF-IDF overweights the short title with very strong term matches (*“Information Theory”, “Machine Learning”*). The title is too general and does not explicitly explain the connection to learning targets, so it is labeled 0. This shows a limitation of purely lexical similarity for semantic questions.

---

#### Example 9 – Q4: Uncertainty and probabilistic reasoning (TF-IDF)

- **Question (q4):**  
  *“How are uncertainty or probabilistic reasoning used or interpreted in the models or methods described?”*

- **Top-1 chunk (incorrect, label = 0):**  
  `1811.11669v1_title_0000`  
  > “TOWARDS IDENTIFYING AND MANAGING SOURCES OF UNCERTAINTY IN AI AND MACHINE LEARNING MODELS - AN OVERVIEW”

- **Relevant chunks (label = 1) ranked lower:**  
  - `1811.11669v1_introduction_0002`  
  - `1811.11669v1_introduction_0003`  
  - `1811.11669v1_abstract_0001`  

  These chunks actually describe different **sources of uncertainty**, and how they are quantified and managed in AI/ML models.

- **Why this is a failure:**  
  TF-IDF again prefers the very keyword-dense title over explanatory prose. The title contains “uncertainty” and “machine learning models” and thus gets the highest score, even though the detailed explanation of uncertainty handling is in other chunks.

---

#### Example 10 – Q5: Benchmarks and evaluation setups (TF-IDF)

- **Question (q5):**  
  *“What kinds of benchmarks or evaluation setups are used or discussed for assessing model performance?”*

- **Top-1 chunk (incorrect, label = 0):**  
  `2110.12773v1_title_0000`  
  > “Scientific Machine Learning Benchmarks”

- **Relevant chunks (label = 1) appear lower:**  
  - `2006.15680v1_abstract_0002` – discusses evaluation using F1 and convex hull of the training set.  
  - `1907.07543v1_introduction_0008` – compares deep transfer learning and classical ML under specific data regimes.

- **Why this is a failure:**  
  TF-IDF focuses on the term “Benchmarks” in the title and gives it a very high similarity score (≈0.26), but the title alone does not describe specific benchmark setups. More detailed paragraphs about evaluation metrics and experimental setups are ranked lower, even though they are labeled as relevant.

---

## Conclusion

On this dataset:

- Both rule-based methods (keyword overlap and TF-IDF cosine) achieve reasonable performance and validate the quality of the labeled data.
- The **keyword overlap** baseline is simple and interpretable, with strong **recall** but many false positives.
- The **TF-IDF** baseline slightly improves overall accuracy and provides noticeably better **Precision@5 / Recall@5**, making it a stronger choice for retrieving top-k chunks in a RAG pipeline.
- Qualitative examples show consistent patterns:
  - Titles with strong lexical overlap are often over-ranked (leading to false positives).
  - Detailed explanatory chunks may be under-ranked when they use more varied vocabulary.
  - Neither baseline can fully capture semantic relations beyond surface word overlap.

All scripts, metrics, qualitative examples, and this analysis were implemented by **Mohammadali Dehghani (Member 2)** of the QueryMinds group.
