## Observed Failure Modes (Strict Constrained RAG 2)

### Example 1

**Question ID:** q1  
**Question:** _What problem, limitation, or research gap motivates a machine learning approach?_

**Observed issues:**
- Fabricated limitations (training data availability, sensitivity) not present in retrieved chunks
- Heavy repetition of identical sentences
- Irrelevant conclusion about ML field size and activity
- Unsupported reference (“Bartlett”)

**Strength:**
- Attempts to frame limitations at a high, conceptual level

**Failure types:**
- Fabrication of limitations  
- Question misalignment  
- Redundancy  
- Unsupported citation  

---

### Example 2

**Question ID:** q2  
**Question:** _What is the central idea or key contribution of a machine learning approach?_

**Observed issues:**
- Introduces a fully specified **model entropy formula** not defined in retrieved text
- Reframes a conceptual conjecture as a concrete optimization method
- Adds procedural elements (data augmentation) absent from sources

**Strength:**
- Uses terminology present in retrieved chunks (model entropy, intrinsic objectives)

**Failure types:**
- Over-specification  
- Fabrication of formal definitions  
- Concept-to-method inflation  

---

### Example 3

**Question ID:** q3  
**Question:** _How are information-theoretic concepts connected to learning objectives?_

**Observed issues:**
- Generic, textbook-style explanation of entropy and KL divergence
- Repetition of definitions without linking to learning target selection
- Weak grounding in the specific conjectures discussed in sources

**Strength:**
- Correct high-level definitions of information-theoretic terms

**Failure types:**
- Overgeneralization  
- Weak context grounding  
- Redundancy  

---

### Example 4

**Question ID:** q4  
**Question:** _How do machine learning methods represent and use uncertainty?_

**Observed issues:**
- Incorrect attribution of uncertainty handling to backpropagation
- Ignores retrieved taxonomy of uncertainty (model fit, data quality, scope)
- Drifts into generic predictive modeling explanation

**Strength:**
- Acknowledges probabilistic modeling at a high level

**Failure types:**
- Conceptual confusion  
- Method misattribution  
- Context underutilization  

---

### Example 5

**Question ID:** q5  
**Question:** _What evaluation protocols and benchmark setups are commonly used?_

**Observed issues:**
- Reframes benchmark-oriented question into generic ML evaluation workflow
- Ignores scientific ML benchmarking, curated datasets, and community benchmarks
- No mention of reproducibility or architecture comparison

**Strength:**
- Mentions standard evaluation practices (CV, holdout)

**Failure types:**
- Question scope shift  
- Overgeneralization  
- Benchmark context omission  

---
