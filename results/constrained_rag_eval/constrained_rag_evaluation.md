
## Observed Failure Modes (Strict Constrained RAG)


### Example 1

**Question ID:** q1  
**Question:** _What problem, limitation, or research gap motivates a machine learning approach?_

**Observed hallucination:**  
The generated answer introduces claims about **training data availability**, **data quality sensitivity**, and presents a broad, generic discussion about the size and activity of the machine learning field. None of these points are explicitly stated in the retrieved chunks, which instead focus on **quantitative analysis of ML research topics and dataset characteristics**.

Additionally, the answer contains **heavy repetition**, an unrelated “Conclusion” section, and an incomplete reference (“Bartlett”), none of which are grounded in the retrieved context.

**Type of hallucination:**
- Fabrication of limitations  
- Question misalignment  
- Redundancy and incoherence  
- Unsupported references  

---

### Example 2

**Question ID:** q2  
**Question:** _What is the central idea or key contribution of a machine learning approach?_

**Observed hallucination:**  
The answer introduces a **formal definition and equation for a “model entropy function”**, describing it as a sum of learning loss and internal learning loss. While the retrieved chunks discuss *model entropy conceptually*, they **do not define a mathematical formula**, nor frame it as a concrete data augmentation procedure.

This represents a shift from a **conceptual conjecture** in the source material to a **fully specified method**, which is not supported by the retrieved text.

**Type of hallucination:**
- Over-specification  
- Fabrication of formal definitions  
- Inflation of conceptual ideas into methods  

---

### Example 3

**Question ID:** q3  
**Question:** _How are information-theoretic concepts (e.g., entropy, mutual information, KL divergence) connected to learning objectives or targets?_

**Observed hallucination:**  
The answer provides a **generic textbook-style explanation** of entropy and KL divergence, with repeated statements and vague claims about “information consumption” and “information content of the system.”

However, it fails to connect these concepts to **learning target selection**, **empirical similarity measures**, or the **specific conjectures discussed in the retrieved chunks**. The response largely ignores the concrete framing present in the source material.

**Type of hallucination:**
- Overgeneralization  
- Weak grounding in retrieved context  
- Redundant explanations  

---

### Example 4

**Question ID:** q4  
**Question:** _How do machine learning methods represent and use uncertainty (e.g., probabilistic modeling, Bayesian methods, predictive uncertainty)?_

**Observed hallucination:**  
The answer incorrectly emphasizes **backpropagation** as a mechanism for representing uncertainty and provides a generic explanation of predictive modeling. Backpropagation is a **training algorithm**, not an uncertainty modeling technique.

The retrieved chunks instead discuss **sources of uncertainty**, such as model fit, data quality, and scope compliance, as well as design-time and runtime uncertainty management—none of which are properly reflected in the generated answer.

**Type of hallucination:**
- Conceptual confusion  
- Incorrect attribution of methods  
- Context underutilization  

---

### Example 5

**Question ID:** q5  
**Question:** _What evaluation protocols and benchmark setups are commonly used to assess model performance?_

**Observed hallucination:**  
The answer focuses on **standard ML evaluation procedures** (cross-validation, holdout, hyperparameter tuning) but ignores the retrieved chunks’ emphasis on **scientific machine learning benchmarking**, including curated benchmark suites, community-level benchmarks, architectural comparisons, and reproducibility-driven evaluation.

As a result, the answer reframes a **benchmark-focused question** into a **generic ML evaluation explanation**, missing the central theme of the retrieved context.

**Type of hallucination:**
- Question scope shift  
- Overgeneralization  
- Failure to extract benchmark-specific information  

---