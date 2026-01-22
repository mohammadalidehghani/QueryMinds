import json
from pathlib import Path
from typing import Optional, List

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
from pydantic import PrivateAttr

from langchain_community.vectorstores import FAISS
from langchain_core.prompts import PromptTemplate
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_core.language_models.llms import LLM


# ============================================================
# Local HuggingFace LLM wrapper (IDENTICAL to Task 1)
# ============================================================

class LocalHFLLM(LLM):
    _pipe: any = PrivateAttr()

    def __init__(self, pipe):
        super().__init__()
        self._pipe = pipe

    @property
    def _llm_type(self) -> str:
        return "local_huggingface"

    def _call(self, prompt: str, stop: Optional[List[str]] = None) -> str:
        out = self._pipe(
            prompt,
            max_new_tokens=256,
            do_sample=True,
            temperature=0.7,
        )
        return out[0]["generated_text"][len(prompt):]


# ============================================================
# PATHS & CONFIG
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

DATA_DIR = PROJECT_ROOT / "data"
RESULTS_DIR = PROJECT_ROOT / "results"
RESULTS_DIR.mkdir(exist_ok=True)

FAISS_INDEX_PATH = DATA_DIR / "faiss_index_v2"
QUESTIONS_PATH = DATA_DIR / "questions_v2.json"
OUTPUT_PATH = RESULTS_DIR / "constrained_rag_results_v2.json"

TOP_K = 5

REFUSAL_TEXT = "The answer is not found in the provided documents."


# ============================================================
# LOAD QUESTIONS
# ============================================================

def load_questions(path: Path):
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


questions_data = load_questions(QUESTIONS_PATH)
questions = [(q["id"], q["question"]) for q in questions_data]


# ============================================================
# EMBEDDINGS & RETRIEVER (IDENTICAL to Task 1)
# ============================================================

embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

vectorstore = FAISS.load_local(
    FAISS_INDEX_PATH,
    embeddings,
    allow_dangerous_deserialization=True
)

retriever = vectorstore.as_retriever(
    search_type="similarity",
    search_kwargs={"k": TOP_K}
)


# ============================================================
# STRICT CONSTRAINED PROMPT (TASK 2 CORE)
# ============================================================

prompt = PromptTemplate(
    input_variables=["context", "question"],
    template="""
You are a question-answering system.

You must follow these rules strictly:
- Use ONLY the information provided in the CONTEXT.
- Do NOT use any external or prior knowledge.
- Do NOT infer, assume, generalize, or combine information from multiple parts of the CONTEXT.
- Do NOT provide background, motivation, or explanations not explicitly stated.
- Answer only the QUESTION, using factual statements found verbatim in the CONTEXT.
- If the answer is not explicitly stated in the CONTEXT, respond with:
  "The answer is not found in the provided documents."

CONTEXT:
{context}

QUESTION:
{question}

ANSWER:
"""
)


# ============================================================
# LOAD LLM (IDENTICAL to Task 1)
# ============================================================

MODEL_NAME = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    torch_dtype=torch.float32
)

text_gen_pipeline = pipeline(
    "text-generation",
    model=model,
    tokenizer=tokenizer
)

llm = LocalHFLLM(text_gen_pipeline)


# ============================================================
# CONSTRAINED RAG PIPELINE (TASK 2)
# ============================================================

def constrained_rag(question: str):
    docs = retriever.invoke(question)

    # If no retrieved chunks -> immediate refusal
    if len(docs) == 0:
        return {
            "question": question,
            "answer": REFUSAL_TEXT,
            "retrieved_chunks": []
        }

    context = "\n\n".join(doc.page_content for doc in docs)

    formatted_prompt = prompt.format(
        context=context,
        question=question
    )

    answer_text = llm.invoke(formatted_prompt).strip()

    # Programmatic refusal logic
    if (
        answer_text == ""
        or "not found" in answer_text.lower()
        or "cannot find" in answer_text.lower()
        or "i don't know" in answer_text.lower()
    ):
        answer_text = REFUSAL_TEXT

    return {
        "question": question,
        "answer": answer_text,
        "retrieved_chunks": [doc.page_content for doc in docs]
    }


# ============================================================
# RUN PIPELINE
# ============================================================

def main():
    results = []

    for qid, qtext in questions:
        print(f"Processing {qid}")
        res = constrained_rag(qtext)
        res["question_id"] = qid
        results.append(res)

    with OUTPUT_PATH.open("w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    print(f"\nSaved constrained RAG results to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
