import json
from pathlib import Path

from sentence_transformers import SentenceTransformer, InputExample, losses
from torch.utils.data import DataLoader


# ------------------------------------------------------------------
# Paths
# ------------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
DATA_DIR = PROJECT_ROOT / "data"
MODEL_OUT_DIR = PROJECT_ROOT / "models" / "finetuned-minilm"
MODEL_OUT_DIR.mkdir(parents=True, exist_ok=True)

ML_PAIRS_PATH = DATA_DIR / "ml_pairs" / "ml_pairs_v3.jsonl"


# ------------------------------------------------------------------
# Load training pairs (ONLY label = 1)
# ------------------------------------------------------------------

def load_positive_pairs(path: Path):
    examples = []

    with path.open("r", encoding="utf-8") as f:
        for line in f:
            obj = json.loads(line)

            if int(obj["label"]) != 1:
                continue  # ignore negatives

            question = obj["question"].strip()
            chunk = obj["chunk"].strip()

            if question and chunk:
                examples.append(
                    InputExample(texts=[question, chunk])
                )

    return examples


# ------------------------------------------------------------------
# Main training
# ------------------------------------------------------------------

def main():
    print("Loading base embedding model...")
    model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

    print("Loading positive training pairs...")
    train_examples = load_positive_pairs(ML_PAIRS_PATH)
    print(f"Loaded {len(train_examples)} positive pairs")

    if len(train_examples) < 2:
        raise ValueError("Not enough training pairs for contrastive learning!")

    train_dataloader = DataLoader(
        train_examples,
        shuffle=True,
        batch_size=16
    )

    train_loss = losses.MultipleNegativesRankingLoss(model)

    print("Starting fine-tuning...")
    model.fit(
        train_objectives=[(train_dataloader, train_loss)],
        epochs=1,
        warmup_steps=100,
        show_progress_bar=True
    )

    print(f"Saving fine-tuned model to {MODEL_OUT_DIR}")
    model.save(str(MODEL_OUT_DIR))

    print("Done.")


if __name__ == "__main__":
    main()
