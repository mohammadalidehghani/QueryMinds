import os
import pandas as pd
from conllu import parse_incr

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
    stats.append({"file": fname, "sentences": n_sent,
                  "tokens": n_tok, "unique_lemmas": n_lemma})

df = pd.DataFrame(stats)
df["tokens_per_sentence"] = df["tokens"] / df["sentences"]
print(df.describe())
df.to_csv("docs/corpus_stats.csv", index=False)
print("---Saved stats to docs/corpus_stats.csv---")
