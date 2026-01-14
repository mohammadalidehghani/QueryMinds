\# Annotation Guidelines for Relevance Labels



\## 1. Core rule



\*\*Label = 1\*\* if someone could read \*only this chunk\* and then write at least \*\*one clear, specific sentence\*\* that answers the question.



\*\*Label = 0\*\* if that’s not possible.  

If the case feels borderline or “maybe”, treat it as \*\*0\*\*.



\## 2. Small clarifications



\- Consistency matters more than perfection.

\- “Clear \& specific” means the answer is not vague, not guessed, and is based on explicit information in the chunk.

\- Just seeing keywords like “dataset”, “robustness”, or “features” is not enough for label 1.

\- If the chunk is mostly general background and not really needed to answer the question, label it 0.

\- If you’re unsure whether it truly answers the question, treat it as 0.



