import os
import pandas as pd
from conllu import parse_incr
import matplotlib.pyplot as plt

input_dir = "data/conllu"
stats = []

for fname in os.listdir(input_dir):
    if not fname.endswith(".conllu"):
        continue
    n_sent = 0
    n_tok = 0
    n_lemma = 0
    with open(os.path.join(input_dir, fname), encoding="utf8") as f:
        for sent in parse_incr(f):
            n_sent += 1
            n_tok += len(sent)
            n_lemma += len(set(t["lemma"] for t in sent if t["lemma"]))
    stats.append({
        "file": fname,
        "sentences": n_sent,
        "tokens": n_tok,
        "unique_lemmas": n_lemma
    })

df = pd.DataFrame(stats)
df["tokens_per_sentence"] = df["tokens"] / df["sentences"]

os.makedirs("docs", exist_ok=True)

summary = df.describe()
summary_path = "docs/corpus_summary.txt"
with open(summary_path, "w", encoding="utf8") as f:
    f.write(str(summary))
print(f"--- Saved summary to {summary_path} ---")

csv_path = "docs/corpus_stats.csv"
if os.path.exists(csv_path):
    os.remove(csv_path)
df.to_csv(csv_path, index=False)
print(f"--- Saved stats to {csv_path} ---")

plt.figure(figsize=(10, 6))
plt.hist(df["tokens"], bins=30, edgecolor='black')
plt.title("Distribution of Number of Tokens per Document")
plt.xlabel("Number of Tokens")
plt.ylabel("Number of Documents")
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig("docs/token_distribution.png", dpi=300)
print(f"--- Saved plot to docs/token_distribution.png---")


plt.figure(figsize=(10, 6))
plt.hist(df["sentences"], bins=30, edgecolor='black')
plt.title("Distribution of Number of Sentences per Document")
plt.xlabel("Number of Sentences")
plt.ylabel("Number of Documents")
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig("docs/sentence_distribution.png", dpi=300)
print("--- Saved plot to docs/sentence_distribution.png ---")


plt.figure(figsize=(8, 6))
plt.scatter(df["tokens"], df["unique_lemmas"], alpha=0.7)
plt.title("Tokens vs. Unique Lemmas per Document")
plt.xlabel("Total Tokens")
plt.ylabel("Unique Lemmas")
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig("docs/tokens_vs_lemmas.png", dpi=300)
print("--- Saved plot to docs/tokens_vs_lemmas.png ---")
