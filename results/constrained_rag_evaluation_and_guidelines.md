# Guidelines and Evaluation – Strict Constrained RAG

---

## Relation to Milestone 2 and Vanilla RAG

The Strict Constrained RAG pipeline builds directly on the Vanilla RAG setup introduced in Milestone 3 and reuses the same embedding-based retrieval mechanism from Milestone 2.

To ensure a fair comparison, no changes are made to the retrieval model, embeddings, FAISS index, or document chunks.  
The only modification is applied at the generation stage, where explicit constraints are introduced to limit hallucinations.

The constrained RAG pipeline can be executed via the script:


---

## Retrieval Setup (Unchanged)

The retrieval component of the constrained RAG pipeline is identical to that of Vanilla RAG:

- **Model**: `sentence-transformers/all-MiniLM-L6-v2`
- **Representation**: dense sentence embeddings
- **Similarity measure**: cosine similarity
- **Retrieval mechanism**: FAISS approximate nearest neighbor search
- **Top-k**: 5 retrieved chunks per question

This design choice isolates the effect of generation constraints, ensuring that any observed differences in output quality stem from the constrained prompting strategy rather than retrieval changes.

---

## Strict Constrained RAG Pipeline

The Strict Constrained RAG pipeline follows the same high-level structure as Vanilla RAG, with an additional constraint layer applied to answer generation:

1. Encode the input question using the same embedding model as in Milestone 2.
2. Retrieve the top-k most similar document chunks using FAISS similarity search.
3. Concatenate retrieved chunks into a single context block.
4. Pass the context and question to a local generative language model with a strictly constrained prompt.
5. Apply a refusal mechanism if the answer is not explicitly supported by the retrieved context.

Unlike Vanilla RAG, the constrained pipeline explicitly instructs the model to refuse answering when the requested information is not directly stated in the retrieved documents.

---

## Prompt Constraints

The constrained RAG prompt enforces the following rules:

- The model may use **only the provided context**.
- External knowledge, inference, generalization, or background explanations are prohibited.
- The model must **not repeat or reference the context explicitly**.
- If the answer is not explicitly stated in the context, the model must respond with a standardized refusal message:

> *“The answer is not found in the provided documents.”*

This prompt-based restriction aims to reduce hallucinations by limiting the model’s generative freedom.

---

## Language Model

- **Model**: `TinyLlama/TinyLlama-1.1B-Chat-v1.0`
- **Execution**: fully local
- **Role**: answer generation only

The language model remains unchanged from the Vanilla RAG setup, ensuring that improvements or failures can be attributed to the applied constraints rather than model differences.

---

## Observed Failure Modes (Strict Constrained RAG)

While the constrained setup reduces certain types of hallucinations, the evaluation revealed several remaining limitations:

- **Semantic Synthesis**  
  The model still attempts to merge information from multiple retrieved chunks into a single coherent explanation, even when the answer is not explicitly stated.

- **Implicit Inference**  
  In some cases, answers are constructed by paraphrasing or extrapolating from related context rather than directly quoting or extracting factual statements.

- **Prompt Leakage**  
  The model occasionally reproduces fragments of the prompt or context markers (e.g., “CONTEXT:”), indicating incomplete separation between instruction and generation.

- **Overly General Answers**  
  Some responses revert to generic machine learning knowledge that is thematically related but not grounded in the retrieved chunks.

These behaviors demonstrate that prompt-based constraints alone are insufficient to fully eliminate hallucinations.

---

## Examples of Residual Hallucinations in Constrained RAG Output

### Example 1

**Question ID:** q1  
**Question:** *What main problem or gap in existing machine learning methods is being addressed?*

**Observed behavior:**  
The model correctly identifies the “curse of dimensionality” but expands the answer with explanatory material that is not explicitly stated in the retrieved context and partially reproduces prompt artifacts.

**Type of issue:**
- Implicit inference  
- Prompt leakage

---

### Example 2

**Question ID:** q2  
**Question:** *What is the central contribution or main idea of the approach being described?*

**Observed behavior:**  
The answer synthesizes a high-level interpretation of socio-structural explanations without clearly anchoring each claim to explicit statements in the retrieved chunks.

**Type of issue:**
- Contextual overgeneralization  
- Semantic synthesis

---

### Example 3

**Question ID:** q8  
**Question:** *In what ways are machine learning methods applied to concrete scientific or engineering problems?*

**Observed behavior:**  
The model produces a broad list of generic ML applications (e.g., classification, NLP, robotics), despite such enumerations not being explicitly present in the retrieved context.

**Type of issue:**
- Background knowledge injection

---

## Summary of Findings

The Strict Constrained RAG setup demonstrates a clear reduction in unconstrained hallucinations compared to Vanilla RAG.  
However, the results also reveal that prompt-level constraints alone cannot fully prevent semantic inference and synthesis in generative language models.