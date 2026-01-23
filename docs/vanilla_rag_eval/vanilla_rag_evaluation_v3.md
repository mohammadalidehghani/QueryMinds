---

## Observed Failure Modes (Vanilla RAG v3)

The evaluation of the Vanilla RAG outputs revealed several recurring issues:

---


## Examples of Hallucinations Observed in Vanilla RAG Output

### Example 1:

**Question ID:** q1  
**Question:** _What problem, limitation, or research gap motivates a machine learning approach?_

**Observed hallucination:**  
The generated answer completely **misinterprets the question intent**. Instead of identifying a problem, limitation, or research gap that motivates the use of machine learning, the answer reframes the task as a **summary request** (“Can you summarize the findings of the study”) and proceeds to list popular machine learning topics.

The response focuses on **enumerating machine learning techniques** (e.g., Deep Learning, Reinforcement Learning, NLP, Computer Vision) rather than explaining *why* a machine learning approach is necessary or what limitation in existing methods motivates it.  
Additionally, the answer contains **redundant repetitions** (e.g., Reinforcement Learning, Computer Vision, NLP listed multiple times) and an **incomplete list**, indicating weak answer coherence.

While the retrieved chunks discuss a quantitative analysis of machine learning research topics, the answer fails to abstract from this context to address the conceptual motivation behind adopting machine learning.

**Type of hallucination:**

- Severe question misinterpretation  
- Answer drift  
- Redundancy and incoherence  
- Failure to extract motivation from context

---

### Example 2:
**Question ID:** q2  
**Question:** _What is the central idea or key contribution of a machine learning approach?_

**Observed hallucination:**  
The generated answer does not provide an answer at all, but instead **rephrases the question into another question**. Rather than stating the central idea or key contribution, it asks about the role of model entropy and its relation to traditional optimization-based views of machine learning.

This represents a **failure to generate content grounded in the retrieved context**. While the retrieved chunks do discuss model entropy, intrinsic learning, and a critique of purely optimization-driven definitions of machine learning, the answer never synthesizes these ideas into a **clear, declarative statement of contribution** (e.g., proposing model entropy as a new learning objective or redefining learning as intrinsically multiobjective).

As a result, the response neither summarizes nor extracts the key contribution, and instead shifts responsibility back to the reader.

**Type of hallucination:**

- Answer omission  
- Question echoing / rephrasing  
- Failure to synthesize retrieved context

---

### Example 3:

**Question ID:** q7  
**Question:** _How does theoretical analysis (e.g., proofs or bounds) relate to empirical evidence (e.g., experiments or ablations)?_

**Observed hallucination:**  
The generated answer provides a **generic, high-level discussion** about the relationship between theoretical analysis and empirical evidence in machine learning, without grounding this discussion in the **specific retrieved context**.

While the answer introduces the concept of combining theory and experiments and mentions a proposed framework (“mathematical-principle-based machine”), it does not clearly explain **how theoretical results (e.g., proofs, bounds)** are validated, complemented, or contrasted with **empirical evidence (e.g., experiments, cross-validation, ablations)** in practice.  
Instead, it remains abstract and philosophical, offering general statements that could apply to almost any machine learning paper.

Although some retrieved chunks touch on theoretical perspectives and empirical challenges, the answer does not synthesize these into a **concrete explanation of interaction between theory and experiments**, such as theory guiding model design and experiments validating assumptions or revealing gaps.

As a result, the response only loosely addresses the question and lacks specificity.

**Type of hallucination:**

- Overgeneralization  
- Answer abstraction  
- Weak grounding in retrieved context

---

### Example 4:

**Question ID:** q12  
**Question:** _What fairness or bias risks arise in machine learning systems, and what mitigation approaches are used?_

**Observed hallucination:**  
The generated answer provides a **broad ethical and philosophical discussion** about the ethics of machine learning, but it does not explicitly identify **concrete fairness or bias risks**, nor does it describe **specific mitigation approaches**.

Instead of addressing issues such as algorithmic bias, discrimination, calibration trade-offs, or mitigation techniques (e.g., fairness constraints, reweighting, post-processing, governance practices), the answer remains at the level of **normative reflection** on ethical accountability and relational ethics.  
Although the retrieved chunks discuss fairness concepts (e.g., actuarial fairness vs. solidarity, responsibility, aggregate–individual tensions), the answer does not translate these into **operational risks** or **actionable mitigation strategies** used in machine learning practice.

As a result, the response reads more like an **introductory ethics position paper summary** than a direct answer to the question.

**Type of hallucination:**

- Question under-specification  
- Answer abstraction  
- Missing mitigation strategies  
- Weak operational grounding

---

### Example 5:

**Question ID:** q15  
**Question:** _Which modeling and architectural design choices are especially important, and why?_

**Observed hallucination:**  
The generated answer largely **repeats and paraphrases the retrieved context** without synthesizing it into a clear, question-focused response. While it discusses modeling perspectives, human intuition, model topology, and internal representations, it does not explicitly identify **which specific modeling or architectural design choices are important** nor clearly explain **why** they matter in practice.

The answer remains descriptive and philosophical, focusing on conceptual viewpoints (e.g., modeling perspective, internal model representations) rather than extracting actionable design choices such as:
- architectural depth vs. shallow models,
- structural properties of models (e.g., sparsity, modularity),
- interpretability vs. complexity trade-offs,
- internal representations beyond decision boundaries.

As a result, the response mirrors the retrieved chunks closely but fails to **abstract and prioritize** key architectural decisions in a concise and evaluative manner.

**Type of hallucination:**

- Answer verbosity without synthesis  
- Weak prioritization of key design choices  
- Question under-addressing  
- Context paraphrasing instead of reasoning