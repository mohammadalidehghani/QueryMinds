## Examples of Errors Observed (Constrained RAG 3)

### Example 1

**Question ID:** q1  
**Question:** _What problem, limitation, or research gap motivates a machine learning approach?_

**Observed issues:**
- Answer repeatedly **rephrases the question into new sub-questions**
- Introduces speculative claims about “novel techniques not yet developed”
- Does not extract a concrete research gap from retrieved chunks
- Drifts into generic practitioner-oriented statements

**Strength:**
- Mentions dataset size and extrapolation, which appear in retrieved context

**Failure types:**
- Question echoing  
- Answer drift  
- Weak grounding in retrieved evidence  

---

### Example 2

**Question ID:** q2  
**Question:** _What is the central idea or key contribution of a machine learning approach?_

**Observed issues:**
- Answer collapses into a **tautology** (“the key contribution is learning”)
- Heavy repetition of identical sentences
- No reference to the quantitative methodology or holistic field analysis described in retrieved chunks

**Strength:**
- Correctly frames machine learning as experience-driven improvement

**Failure types:**
- Missing core contribution  
- Redundancy  
- Oversimplification  

---

### Example 3

**Question ID:** q3  
**Question:** _How are information-theoretic concepts connected to learning objectives or targets?_

**Observed issues:**
- Introduces incomplete or incorrect associations (e.g., mutual information as similarity)
- Answer is truncated and unfinished
- Fails to clearly connect concepts to **learning target selection**, which is central in retrieved text

**Strength:**
- Mentions model entropy and information-theoretic measures present in sources

**Failure types:**
- Conceptual imprecision  
- Incompleteness  
- Partial context utilization  

---

### Example 4

**Question ID:** q4  
**Question:** _How do machine learning methods represent and use uncertainty?_

**Observed issues:**
- Provides a generic probabilistic description detached from retrieved uncertainty taxonomy
- Ignores distinctions between model fit, data quality, and scope compliance
- Answer is cut off mid-list

**Strength:**
- Acknowledges probabilistic and Bayesian approaches

**Failure types:**
- Overgeneralization  
- Context underutilization  
- Incomplete answer  

---

### Example 5

**Question ID:** q5  
**Question:** _What evaluation protocols and benchmark setups are commonly used to assess model performance?_

**Observed issues:**
- Focuses on standard ML metrics and cross-validation
- Does not reflect the retrieved emphasis on **scientific ML benchmarking**, curated datasets, or community benchmarks
- Answer is truncated

**Strength:**
- Mentions common evaluation metrics used in practice

**Failure types:**
- Question scope narrowing  
- Benchmark context omission  
- Incompleteness  

---
