# Task 3 — Chunking analysis + retrieval impact (write-up)

## 1) Quantitative impact (using the 15×20 human labels)

We evaluate retrieval with:
- **P@5**: how many of the top-5 retrieved chunks are relevant (average over questions)
- **R@5 within top-20**: among the relevant chunks that exist in the 20 candidates, what fraction appear in the top-5

**Overall (15 questions):**
- Old (re-labeled old chunks): **Mean P@5 = 0.093**, **Mean R@5 = 0.144**
- v2 (new chunking): **Mean P@5 = 0.267**, **Mean R@5 = 0.205**
- Improvement: **ΔP@5 = +0.173**, **ΔR@5 = +0.060**

Interpretation in simple words:
- With v2, the system is much more likely to put a relevant chunk **inside the top 5**.

## 2) What “bad chunks” mean in this project

A chunk can be “bad” for retrieval (and for generation) mainly because:
1) **Too short**: it contains only a title or a very small fragment → no usable evidence.
2) **Contextually unclear**: it starts in the middle of a paragraph or uses “this/these/it” without context → hard to match a question.
3) **Too long / too broad**: it mixes several ideas → keyword match becomes noisy and the chunk is not a clean piece of evidence.

## 3) Examples from our data (old vs v2)

### Example A — big improvement (q12)
**Question:** What fairness or bias risks arise in machine learning systems, and what mitigation approaches are used?

- Old: relevant chunks in **top-5 = 1**, in **top-20 = 6**
- v2: relevant chunks in **top-5 = 5**, in **top-20 = 8**

**Old (problematic candidate):**
- rank 4, chunk_id `2306.14624v2_title_0000` (≈7 words)
  - excerpt: “Insights From Insurance for Fair Machine Learning”

**v2 (better evidence in top-5):**
- rank 1, chunk_id `1703.10121v1_abstract_0000` (≈133 words)
  - excerpt: “Which topics of machine learning are most commonly addressed in research ? This question was initially answered in 2007 by doing a qualitative survey among distinguished researchers . In our study , we revisit this question from a quantitative perspective . Concretely , we collect …”

Why v2 is better here:
- v2 surfaces more fairness/bias-related content in the top results, which increases the chance that generation uses correct evidence.

---

### Example B — “contextually unclear” chunk (q11)
**Question:** What methods are used to explain or interpret model behavior (e.g., feature importance, saliency, counterfactual explanations)?

- Old: relevant chunks in **top-5 = 0**, in **top-20 = 5**
- v2: relevant chunks in **top-5 = 2**, in **top-20 = 7**

**Old (context is missing):**
- rank 1, chunk_id `1504.03874v1_introduction_0002` (≈141 words)
  - excerpt: “Introduction In an age of user generated web-contents and of portable devices with embedded computer vision capabilities , machine learning ( ML ) and big data mining questions are fundamental . As a result , these questions naturally penetrate neighboring research fields , including belief …”

**v2 (self-contained interpretability chunk in top-5):**
- rank 5, chunk_id `2409.03632v1_abstract_0000` (≈148 words)
  - excerpt: “What is it to interpret the outputs of an opaque machine learning model ? One approach is to develop interpretable machine learning techniques . These techniques aim to show how machine learning models function by providing either model-centric local or global explanations , which can …”

Why v2 is better here:
- The v2 chunk is easier to match to the question because it is a clearer, standalone piece of text about interpretation.

---

### Example C — “too short” chunk (q3)
**Question:** How are information-theoretic concepts (e.g., entropy, mutual information, KL divergence) connected to learning objectives or targets?

- Old: relevant chunks in **top-5 = 0**, in **top-20 = 1**
- v2: relevant chunks in **top-5 = 1**, in **top-20 = 2**

**Old (too short / title-only):**
- rank 1, chunk_id `1501.04309v1_title_0000` (≈8 words)
  - excerpt: “Information Theory and its Relation to Machine Learning”

**v2 (information-theoretic content appears in top-5):**
- rank 5, chunk_id `1711.01431v1_abstract_0001` (≈47 words)
  - excerpt: “To this end , we suggest a `` model entropy function '' to be defined that quantifies the efficiency of the internal learning processes . It is conjured that the minimization of this model entropy leads to concept formation . Besides philosophical aspects , some …”

Why v2 is better here:
- Title-only chunks add noise. v2 more often retrieves chunks that mention entropy/KL/mutual information, which aligns better with the question.

---

### Example D — a case that got worse (q13)
**Question:** What limitations are commonly identified, and what future directions or open problems follow from them?

- Old: relevant chunks in **top-5 = 4**, in **top-20 = 4**
- v2: relevant chunks in **top-5 = 3**, in **top-20 = 11**

**Old (broad chunk that is not directly about limitations):**
- rank 5, chunk_id `1903.00092v2_introduction_0002` (≈185 words)
  - excerpt: “Introduction Online decision-making problems fundamentally address the issue of dealing with the uncertainty inherently present in the future . In broad terms , these problems can be addressed in two ways . First , a predictive approach like a machine learning algorithm can be used …”

**v2 (a limitations-oriented chunk is still in top-5, but ranking changed overall):**
- rank 2, chunk_id `1711.01431v1_introduction_0006` (≈196 words)
  - excerpt: “Traditional machine learning techniques typically exploit shallow-structured , and often fixed , architectures . Nevertheless , there is a general consensus that the learning of `` higherorder '' concepts is problematic , and that the solution to this issue is somehow connected to deep architectures …”

Simple interpretation:
- v2 increases the number of potentially relevant candidates in top-20, but for this question the top-5 contained slightly fewer labeled-relevant chunks.

## 4) Cconclusion

Overall, the v2 chunking strategy improved retrieval quality substantially: Mean **P@5 increased from 0.093 to 0.267**, and Mean **R@5 (within top-20) increased from 0.144 to 0.205**. Qualitative inspection shows that v2 reduces “too short” and “contextually unclear” chunks, and more often surfaces self-contained, question-aligned evidence within the top-5 results.
