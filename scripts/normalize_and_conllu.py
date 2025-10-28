import os, json, re
import stanza
from stanza.utils.conll import CoNLL

stanza.download('en')
nlp = stanza.Pipeline('en', processors='tokenize,pos,lemma')


in_dir = "data/parsed_tokens"
out_dir = "data/conllu"
os.makedirs(out_dir, exist_ok=True)

for file in os.listdir(in_dir):
    if not file.endswith(".json"): continue
    with open(os.path.join(in_dir, file), "r", encoding="utf-8") as f:
        doc = json.load(f)

    # Join all sentences from all sections
    text = ""
    for section in doc.values():
        for s in section:
            text += s["sentence"].lower().strip() + "\n"

    # Clean punctuation & normalize spacing
    text = re.sub(r'\s+', ' ', text).strip()

    # Lemmatize and POS-tag
    doc_stanza = nlp(text)
    out_path = os.path.join(out_dir, file.replace(".json", ".conllu"))
    CoNLL.write_doc2conll(doc_stanza, out_path)

    print(f"Converted {file} to {out_path}")
