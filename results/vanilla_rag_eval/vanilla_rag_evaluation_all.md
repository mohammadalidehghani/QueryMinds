## Overall Review of Vanilla RAG Variants (v1–v3)

Across all three Vanilla RAG versions, the qualitative evaluation reveals a consistent pattern of limitations that stem from the **unconstrained nature of the generation step** and the lack of explicit mechanisms for enforcing question grounding, answer focus, and factual alignment with retrieved context.

### General Trends Across Versions

All three versions demonstrate that, while embedding-based retrieval is able to surface **topically relevant context**, the generative model frequently fails to transform this context into **question-aligned, concise, and faithful answers**. Instead, the model often defaults to summarization, abstraction, or generic explanations that only partially reflect the intent of the question.

A recurring issue across versions is the **weak coupling between question intent and answer structure**. Even when retrieved chunks contain sufficient information, the generated answers frequently:
- describe what a paper or study does rather than answering *why*, *what*, or *how* as explicitly requested,
- prioritize narrative coherence over factual precision,
- or reproduce the structure and wording of retrieved passages without synthesis.

### Evolution of Failure Modes

- **Vanilla RAG v1** primarily exhibits **hallucinations in the form of fabricated entities, benchmarks, or named concepts**, as well as context mixing across documents. These errors suggest early-stage failures in factual grounding and content control.

- **Vanilla RAG v2** shows a shift away from explicit fabrication toward more subtle errors, including **question misalignment, overgeneralization, and excessive abstraction**. Answers often remain technically plausible but fail to identify specific contributions, evaluation setups, or motivations requested by the question.

- **Vanilla RAG v3** amplifies these issues, with several cases of **severe question misinterpretation, answer omission, and question rephrasing instead of answering**. Responses increasingly drift toward philosophical or ethical commentary, generic ML explanations, or repeated lists, indicating degradation in answer coherence and intent-following.

### Common Failure Patterns

Across all versions, the following failure modes are consistently observed:

- **Question Misalignment:**  
  Answers frequently fail to directly address the explicit scope of the question.

- **Answer Drift and Abstraction:**  
  Responses move toward high-level discussion instead of extracting concrete information from the retrieved context.

- **Overreliance on Retrieved Text:**  
  The model mirrors retrieved chunks without prioritization, synthesis, or evaluation.

- **Missing or Implicit Contributions:**  
  Key ideas, motivations, or methods are often implied rather than clearly stated.

- **Redundancy and Verbosity:**  
  Answers contain repeated concepts, incomplete lists, or unnecessary elaboration.

### Summary Assessment

Overall, the evaluation demonstrates that **Vanilla RAG pipelines—even with high-quality embedding-based retrieval—are insufficient for reliable question answering without additional constraints**. The observed hallucinations are not limited to factual fabrication, but more commonly manifest as **conceptual drift, scope narrowing, abstraction, and weak answer grounding**.

These findings motivate the need for subsequent pipeline improvements, such as:
- stricter prompt structuring,
- answer length and scope constraints,
- question–answer alignment checks,
- or post-generation validation mechanisms.

Such enhancements are necessary to move from topical relevance toward **faithful, question-driven, and context-grounded answers** in later RAG versions.
