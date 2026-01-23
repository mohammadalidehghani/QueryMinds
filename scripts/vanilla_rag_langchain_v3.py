import json
from pathlib import Path
from typing import Optional, List

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
from pydantic import PrivateAttr

from langchain_community.vectorstores import FAISS
from langchain_core.prompts import PromptTemplate
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.language_models.llms import LLM

# Hugging face
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


# Config

PROJECT_ROOT = Path(__file__).resolve().parent.parent

DATA_DIR = PROJECT_ROOT / "data"
RESULTS_DIR = PROJECT_ROOT / "results"
RESULTS_DIR.mkdir(exist_ok=True)

MODEL_DIR = PROJECT_ROOT / "models" / "finetuned-minilm"

FAISS_INDEX_PATH = DATA_DIR / "faiss_index_v3"
QUESTIONS_PATH = DATA_DIR / "questions_v2.json"
OUTPUT_PATH = RESULTS_DIR / "vanilla_rag_results_v3.json"

TOP_K = 5

# Load questions

def load_questions(path: Path):
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


questions_data = load_questions(QUESTIONS_PATH)
questions = [(q["id"], q["question"]) for q in questions_data]

print(len(questions))

# Embeddings

embeddings = HuggingFaceEmbeddings(
    model_name=str(MODEL_DIR)
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


# Vanilla rag- unconstrained

prompt = PromptTemplate(
    input_variables=["context", "question"],
    template="""
You are an AI assistant.

Use the following document excerpts to answer the question.

Context:
{context}

Question:
{question}

Answer in a clear and concise way.
"""
)


MODEL_NAME = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    dtype=torch.float32
)

text_gen_pipeline = pipeline(
    "text-generation",
    model=model,
    tokenizer=tokenizer
)

llm = LocalHFLLM(text_gen_pipeline)




def vanilla_rag(question: str):
    docs = retriever.invoke(question)

    context = "\n\n".join(doc.page_content for doc in docs)

    formatted_prompt = prompt.format(
        context=context,
        question=question
    )

    answer_text = llm.invoke(formatted_prompt)

    return {
        "question": question,
        "answer": answer_text,
        "retrieved_chunks": [doc.page_content for doc in docs]
    }


# Run pipeline

def main():
    results = []

    for qid, qtext in questions:
        print(f"Processing {qid}")
        res = vanilla_rag(qtext)
        res["question_id"] = qid
        results.append(res)

    with OUTPUT_PATH.open("w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    print(f"\nSaved results to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()