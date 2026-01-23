## Overall Evaluation – Strict Constrained RAG

Across all three evaluated runs of the Strict Constrained RAG pipeline, the applied prompt constraints successfully reduce the **most severe forms of hallucination** observed in the Vanilla RAG setup, such as the invention of entirely new datasets, benchmarks, or institutions. However, the qualitative analysis shows that **strict prompting alone is insufficient to guarantee correct, question-aligned, and context-faithful answers**.

A recurring issue across all constrained variants is **question misalignment**. Instead of extracting the specific information requested (e.g., motivation, key contribution, benchmark setup), the model often rephrases the question, collapses into tautological statements, or produces generic textbook-style explanations. This indicates that while constraints limit unsupported generation, they do not enforce *answer intent alignment*.

Another dominant failure mode is **overgeneralization**. Even when restricted to the provided context, the model frequently responds with broad, high-level descriptions of machine learning concepts (e.g., entropy, uncertainty, evaluation protocols) rather than grounding its answers in the **specific framing and emphasis of the retrieved documents**. As a result, answers remain technically plausible but fail to reflect the distinctive claims or perspectives present in the source material.

The constrained setup also exhibits **over-specification and concept inflation**. Conceptual ideas discussed in the retrieved texts (e.g., model entropy, intrinsic objectives) are sometimes incorrectly reformulated as concrete methods, equations, or procedural techniques that are not explicitly defined in the sources. This demonstrates that constraints reduce external hallucinations but do not fully prevent **internal extrapolation beyond textual evidence**.

Additionally, **context underutilization** persists across all constrained runs. Important structural elements in the retrieved chunks—such as taxonomies of uncertainty, distinctions between benchmark types, or learning target selection frameworks—are often ignored in favor of generic narratives. In several cases, answers are also **truncated or repetitive**, suggesting that strict constraints can negatively affect answer completeness and coherence when the model is uncertain.

Overall, the Strict Constrained RAG pipeline represents a clear improvement over Vanilla RAG in terms of **hallucination containment**, but it introduces new limitations related to **answer quality, abstraction level, and intent alignment**. These results indicate that while prompt-based constraints are a necessary step toward trustworthy RAG systems, they must be complemented by additional mechanisms—such as answer structure enforcement, evidence citation, or post-generation validation—to reliably produce high-quality, context-grounded answers.

---

