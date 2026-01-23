import json
from pathlib import Path
from langchain_core.documents import Document
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings


# Config
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
DATA_DIR = PROJECT_ROOT / "data"

CHUNKS_PATH = DATA_DIR / "chunks" / "chunks_30.jsonl"
FAISS_DIR = DATA_DIR / "faiss_indices" / "faiss_index"


# Load chunks
def load_chunks(path: Path):
    chunks = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            obj = json.loads(line)
            chunks.append(obj)
    return chunks

# Run Faiss
def main():
    chunks = load_chunks(CHUNKS_PATH)
    print(f"Loaded {len(chunks)} chunks")

    documents = []
    for c in chunks:
        doc = Document(
            page_content=c["text"],
            metadata={
                "chunk_id": c["chunk_id"]
            }
        )
        documents.append(doc)

    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

    vectorstore = FAISS.from_documents(documents, embeddings)

    FAISS_DIR.mkdir(exist_ok=True)
    vectorstore.save_local(FAISS_DIR)

    print(f"Done!")


if __name__ == "__main__":
    main()
