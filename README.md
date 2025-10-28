\# QueryMinds — Milestone 1 (Data → Tokenization)



\## Setup

```bash

conda create -n rag\_env python=3.10 -y

conda activate rag\_env

pip install -r requirements.txt

python - <<PY

import nltk, ssl

try: ssl.\_create\_default\_https\_context = ssl.\_create\_unverified\_context

except: pass

nltk.download('punkt')

PY



