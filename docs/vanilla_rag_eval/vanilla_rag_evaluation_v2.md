---

## Observed Failure Modes (Vanilla RAG v2)

Based on the qualitative analysis of the Vanilla RAG v2 outputs, several recurring failure patterns were identified across different question types:

- **Question Misalignment:**  
  Answers frequently fail to directly address the intent of the question. Instead of focusing on *why*, *what*, or *how* aspects explicitly asked, responses often drift toward describing what a study did or summarizing background material.

- **Answer Drift Toward Retrieved Content:**  
  Generated answers tend to follow the narrative of retrieved chunks too closely, even when those chunks are only partially relevant. This leads to responses that mirror document structure rather than extracting question-specific information.

- **Overgeneralization and Abstraction:**  
  In several cases, concrete questions (e.g., evaluation protocols, benchmark setups) are answered with overly generic descriptions of standard machine learning workflows, omitting specific practices highlighted in the retrieved context.

- **Over-Specification of Single Sources:**  
  Some answers focus narrowly on a single dataset, method, or toolkit, presenting excessive detail while failing to generalize toward broader categories or common practices requested by the question.

- **Vague or Implicit Contributions:**  
  When asked about central ideas or key contributions, answers often remain conceptual and abstract, without clearly stating a distinct contribution, novelty, or measurable outcome.

- **Partial Context Utilization:**  
  Relevant elements present in the retrieved chunks (e.g., exploration strategies, intrinsic motivation, benchmarking initiatives) are sometimes omitted, resulting in incomplete or skewed answers.

Overall, these failure modes indicate that **Vanilla RAG v2 lacks strong question grounding and answer constraints**, leading to responses that are loosely related to the question despite relevant retrieved evidence. Such behavior is typical for an unconstrained RAG setup and motivates the need for stricter prompt structuring, answer validation, or constrained generation in later pipeline versions.

---


## Examples of Hallucinations Observed in Vanilla RAG 2 Output

### Example 1:

**Question ID:** q1  
**Question:** _What problem, limitation, or research gap motivates a machine learning approach?_

**Observed hallucination:**  
The generated answer does not directly address the **problem, limitation, or research gap** motivating the machine learning approach.  
Instead, it summarizes conclusions about combining machine learning with Bayesian techniques and shifts focus to a **quantitative survey of machine learning research topics**, including the identification of popular topics and models from 54K papers.

Much of the answer discusses **what the study did** (topic identification, model popularity, survey scope) rather than **why a machine learning approach is needed**.  
This content aligns more closely with the retrieved chunks describing a bibliometric analysis of machine learning literature, rather than answering the conceptual motivation behind adopting machine learning to address a specific limitation or gap.

**Type of hallucination:**

- Question misalignment  
- Answer drift  
- Overgeneralization from retrieved context
---

### Example 2:

### Example 3:

**Question ID:** q2  
**Question:** _What is the central idea or key contribution of a machine learning approach?_

**Observed hallucination:**  
The generated answer presents a **theoretical discussion** on intrinsic learning, model entropy, and meta-cognitive machine learning, but it does not clearly articulate a **single central idea or concrete key contribution** of the machine learning approach.

While the concepts of *model entropy* and *intrinsic multiobjective learning* are introduced, the answer remains largely **conceptual and abstract**, without explicitly stating what the main contribution is (e.g., a new method, framework, definition, or measurable improvement).  
As a result, the response reads more like a **background or conceptual motivation section** rather than a concise summary of the paper’s key contribution.

Additionally, parts of the answer mirror the retrieved chunks’ theoretical framing, but the connection between these ideas and a **distinct, well-defined contribution** is not clearly established.

**Type of hallucination:**

- Answer vagueness  
- Missing core contribution  
- Implicit assumption of novelty


---

### Example 3:

### Example 4:

**Question ID:** q5  
**Question:** _What evaluation protocols and benchmark setups are commonly used to assess model performance?_

**Observed hallucination:**  
The generated answer provides a **generic, high-level description** of a typical machine learning workflow (data preprocessing, model selection, training/validation/testing, reporting), rather than describing **concrete evaluation protocols or benchmark setups** commonly used in practice.

While the retrieved chunks focus specifically on **scientific machine learning benchmarking**, including curated datasets, community benchmarks, architectural comparisons, and standardized evaluation initiatives, the answer does not reference or reflect these aspects. Instead, it reframes the question into a general explanation of how machine learning models are usually evaluated, without mentioning **benchmark datasets, standardized metrics, reproducibility setups, or comparative evaluation frameworks**.

As a result, the answer only loosely aligns with the retrieved context and fails to address the benchmark-oriented nature of the question.

**Type of hallucination:**

- Overgeneralization  
- Context underutilization  
- Answer abstraction


---

### Example 4:

**Question ID:** q6  
**Question:** _What types of datasets or data sources are used for experiments, and what are typical reasons for choosing them?_

**Observed hallucination:**  
The generated answer gives a **very specific and detailed description of a single dataset**, including its alleged origin (UCSD repository), exact number of datasets (109), domain breakdowns, and categorical structure. However, the question asks more generally about **types of datasets or data sources** used for experiments and the **typical reasons for selecting them**, not for an exhaustive description of one particular dataset.

While some retrieved chunks do mention the use of **109 publicly available classification datasets from curated, open-access sources**, the answer extrapolates this into an overly narrow and dataset-specific narrative. It fails to generalize toward broader categories discussed in the retrieved context, such as:
- large-scale experimental datasets from scientific facilities,
- simulated data used due to labeling constraints,
- open, curated benchmark datasets chosen for comparability and reproducibility.

As a result, the answer partially reflects retrieved content but **misses the intent of the question**, which is to summarize common dataset types and motivations rather than document one dataset in isolation.

**Type of hallucination:**

- Over-specification  
- Question scope narrowing  
- Partial context grounding

---

### Example 5:

**Question ID:** q9  
**Question:** _What strategies support learning in complex or changing environments, especially in reinforcement learning (e.g., exploration, adaptation, non-stationarity)?_

**Observed hallucination:**  
The generated answer focuses narrowly on **incremental learning** and the **Dex toolkit**, presenting them as primary strategies for learning in complex or changing environments. While these concepts are mentioned in the retrieved chunks, the question asks more broadly about **general strategies** such as exploration mechanisms, adaptation to non-stationarity, transfer of prior knowledge, or handling sparse rewards.

The answer does not explicitly discuss key reinforcement learning strategies highlighted in the retrieved context, including:
- exploration vs. exploitation trade-offs,
- intrinsic motivation,
- use of prior information or supervised pretraining,
- challenges related to sparse rewards and expensive simulation.

Instead, it reads as a **summary of a specific paper and toolkit**, rather than a generalized overview of commonly used strategies in reinforcement learning for complex or evolving environments.

**Type of hallucination:**

- Question scope narrowing  
- Overemphasis on specific methods  
- Partial context utilization