import sys
import subprocess
import numpy as np
from pathlib import Path
import os
from sentence_transformers import SentenceTransformer, util
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

# ------------------------------------------------------------
# PATHS
# ------------------------------------------------------------
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
DATA_DIR = PROJECT_ROOT / "data"
sys.path.append(str(SCRIPT_DIR))

from rule_based import (
    load_data,
    keyword_overlap_scores,
    tfidf_cosine_scores
)

# ------------------------------------------------------------
# NO PRINTING FROM NOW ON
# ------------------------------------------------------------
class Silent:
    def write(self, *args, **kwargs): pass
    def flush(self): pass

sys.stdout = Silent()
sys.stderr = Silent()

# ------------------------------------------------------------
# BASELINES
# ------------------------------------------------------------
def retrieve_rule_based(question, backend, ctexts):
    if backend == "keyword":
        scores = keyword_overlap_scores([question], ctexts)[0]
    else:
        scores = tfidf_cosine_scores([question], ctexts)[0]
    top_idx = np.argsort(-scores)[:5]
    return [(i, float(scores[i])) for i in top_idx]

def retrieve_embedding(question, ctexts):
    model = SentenceTransformer("all-MiniLM-L6-v2")
    q_emb = model.encode([question], convert_to_tensor=True)
    c_emb = model.encode(ctexts, convert_to_tensor=True)
    sims = util.cos_sim(q_emb, c_emb)[0].cpu().numpy()
    idx = np.argsort(-sims)[:5]
    return [(i, float(sims[i])) for i in idx]

def train_supervised(qtexts, ctexts, labels_dict, q_ids, chunk_ids):
    X, y = [], []
    for qi, q in enumerate(qtexts):
        for ci, c in enumerate(ctexts):
            key = (q_ids[qi], chunk_ids[ci])
            y.append(labels_dict.get(key, 0))
            X.append(q + " [SEP] " + c)

    vec = TfidfVectorizer(stop_words="english", max_features=50000)
    X_vec = vec.fit_transform(X)

    clf = LogisticRegression(max_iter=5000, class_weight="balanced")
    clf.fit(X_vec, y)

    return clf, vec

def retrieve_supervised(question, ctexts, clf, vec):
    X = vec.transform([question + " [SEP] " + c for c in ctexts])
    scores = clf.decision_function(X)
    idx = np.argsort(-scores)[:5]
    return [(i, float(scores[i])) for i in idx]

# ------------------------------------------------------------
# MAIN
# ------------------------------------------------------------
def main():

    clabels, qtexts, ctexts, q_ids, chunk_ids = load_data(DATA_DIR)
    labels_dict = {(q, c): lab for (q, c, lab) in clabels}

    clf, vec = train_supervised(qtexts, ctexts, labels_dict, q_ids, chunk_ids)

    backends = ["keyword", "tfidf", "embedding", "supervised"]
    for backend in backends:
        for q in qtexts:
            if backend == "keyword":
                retrieve_rule_based(q, "keyword", ctexts)
            elif backend == "tfidf":
                retrieve_rule_based(q, "tfidf", ctexts)
            elif backend == "embedding":
                retrieve_embedding(q, ctexts)
            else:
                retrieve_supervised(q, ctexts, clf, vec)

    # -------- Run eval silently --------
    eval_script = PROJECT_ROOT / "scripts" / "eval_rule_baselines.py"
    subprocess.run([sys.executable, str(eval_script)],
                   stdout=subprocess.DEVNULL,
                   stderr=subprocess.DEVNULL)

    # restore printing
    sys.stdout = sys.__stdout__
    print("Done.")

if __name__ == "__main__":
    main()
