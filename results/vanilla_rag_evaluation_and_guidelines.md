# Guidelines and Evaluation

---

## Relation to Milestone 2

The Vanilla RAG pipeline **reuses the embedding-based retrieval baseline** described in Milestone 2.

In Milestone 3, **only the embedding-based baseline is integrated into the Vanilla RAG pipeline**.

To run the Vanilla RAG, simply run the `scripts\vanilla_rag_langchain.py` script.

---

### Embedding-Based Retrieval

The retriever in the Vanilla RAG pipeline is based on the same embedding approach as in Milestone 2:

- **Model**: `sentence-transformers/all-MiniLM-L6-v2`
- **Representation**: dense sentence embeddings
- **Similarity measure**: cosine similarity
- **Retrieval mechanism**: FAISS approximate nearest neighbor search

FAISS is used **only as an efficiency optimization** and does not alter the underlying retrieval logic.

---

## Vanilla RAG Pipeline

The Vanilla RAG pipeline consists of the following steps:

1. Encode the input question using the same embedding model as in Milestone 2.
2. Retrieve the top-_k_ most similar document chunks using FAISS similarity search.
3. Concatenate retrieved chunks into a context block.
4. Pass the context and question to a local generative language model.
5. Generate a free-form answer without additional constraints.

This design intentionally keeps the generation step unconstrained to expose potential hallucinations.

---

## Language Model

- **Model**: `TinyLlama/TinyLlama-1.1B-Chat-v1.0`
- **Execution**: fully local
- **Purpose**: answer generation only

The language model does not influence which documents are retrieved.

---

## Observed Failure Modes (Vanilla RAG)

The evaluation of the Vanilla RAG outputs revealed several recurring issues:

- **Context Mixing:**  
  Answers often combine information from multiple retrieved documents without distinguishing their sources.

- **Hallucinated Content:**  
  Some answers introduce ideas, datasets, or concepts that do not appear in the retrieved context.

- **Verbose and Repetitive Responses:**  
  Answers tend to restate similar ideas multiple times instead of providing concise summaries.

- **Weak Question Grounding:**  
  Certain answers only loosely relate to the original question, despite relevant retrieved chunks.

These issues are characteristic of an unconstrained Vanilla RAG pipeline and are expected at this stage.

---

## Examples of Hallucinations Observed in Vanilla RAG Output

### Example 1:

**Question ID:** q1  
**Question:** _What main problem or gap in existing machine learning methods is being addressed?_

**Observed hallucination:**  
The generated answer claims that the question was “originally asked in 2007 by conducting a qualitative survey” and repeatedly refers to a “study” that revisits this question from a quantitative perspective.

However, the question explicitly asks about the **main problem or gap** in existing machine learning methods, not about the historical origin of the research question.

**Type of hallucination:**

- Context misinterpretation
- Answer drift

---

### Example 2:

**Question ID:** q2  
**Question:** _What is the central contribution or main idea of the approach being described?_

**Observed hallucination:**  
The answer introduces **“Intuition-based engineering (IBE)”** and **“Intuition-based expert engineering (IBEEx)”** as if they were formal, named methodologies and claims that they represent the core contribution of the work.

These concepts **do not appear in the retrieved document excerpts**.

**Type of hallucination:**

- Fabrication of concepts

---

### Example 3:

**Question ID:** q5  
**Question:** _What kinds of benchmarks or evaluation setups are used or discussed for assessing model performance?_

**Observed hallucination:**  
The model refers to a **“European Machine Learning Benchmark (EMLB)”** maintained by the **European Research Council (ERC)** and presents it as a standardized benchmark suite.

Neither the benchmark name nor the institutional association appears in the retrieved chunks.

**Type of hallucination:**

- Fabrication of concepts

---

### Example 4:

**Question ID:** q6  
**Question:** _Which datasets or types of data are used in the experiments, and for what reasons are they chosen?_

**Observed hallucination:**  
Instead of answering the question, the model generates a sequence of **new, unrelated questions**, such as:

- “What is the impact of the chosen target data ...?”
- “What are the limitations of...?”
- “How can we improve...?”
  This indicates that the model partially ignores the original question and drifts into a meta-discussion.

**Type of hallucination:**

- Prompt-following failure
- Question generation instead of answering
- Output derailment

---

### Example 5:

**Question ID:** q4  
**Question:** _How are uncertainty or probabilistic reasoning used or interpreted in the models or methods described?_

**Observed hallucination:**  
The answer centers on **insurance fairness, actuarial reasoning, and “communities of fate”**, despite the retrieved chunks mainly discussing uncertainty in AI models and data-driven components.

Although one retrieved passage references insurance as an analogy, the model elevates this analogy into the main explanation, **overweighting a minor contextual element**.

**Type of hallucination:**

- Thematic drift
- Analogy overextension
