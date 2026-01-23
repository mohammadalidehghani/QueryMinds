# QueryMinds
## Retrieval Augmented Generation for Academic Question Answering

---
QueryMinds Team
- Mohammadali Dehghani 12432957
- Amir Saadati 12434679
- Amina Kadic 12439016
- Meliha Kasapovic 12439367
  

#  Milestone 1 — Overview

Milestone 1 develops a full NLP pipeline that transforms **raw academic PDF documents** into linguistically annotated **CoNLL-U** files.


###  Final Output
A clean, structured corpus of  
 **191 CoNLL-U files**, ready for downstream NLP tasks such as:
- semantic search  
- clustering  
- feature extraction  
- RAG dataset creation  

---
# Milestone 2 — Overview

Milestone 2 extends the Milestone 1 and introduces a complete RAG baseline framework for evaluating retrieval of relevant text segments in the papers.


### Implemented baseline models:
| Model | Description |
|-------|-------------|
| Keyword Overlap | Word-matching heuristic |
| TF–IDF Cosine | Lexical similarity |
| Embedding Similarity | MiniLM-L6-v2 embeddings |
| Supervised Classifier | TF–IDF + Logistic Regression |


---

#  Final Overview

Final version of the projects builds upon the first two milestones, using the existing embedding retrieval for implementing two RAG (Retrieval Augmented Generation) systems:
- **Vanilla RAG**, allowing free-form generation
- **Strict Constrained RAG**, restricting generation to retrieved evidence

## Vanilla RAG Implementation
The system was implemented as a retrieval-augmented generation (RAG) pipeline that combines dense document retrieval with local language model inference.
Scientific documents were embedded using three different embedding models and indexed with FAISS for efficient similarity search. 
For each input question, the top-k most relevant document chunks are retrieved and concatenated into a context.

Answer generation is performed locally using the TinyLlama language model. The retrieved context and the question are inserted into a prompt template and passed to the model without additional constraints in the Vanilla RAG setup. 
The final output consists of the generated answer together with the retrieved document excerpts, enabling qualitative analysis of grounding and hallucinations.

## Constrained RAG Implementation
The Strict Constrained RAG system uses the same embedding-based retrieval pipeline
as the Vanilla RAG setup, relying on dense embeddings indexed with FAISS to retrieve
the top-k most relevant document chunks for each question.

The key difference lies in the generation step.
Answer generation is guided by a strictly constrained prompt that explicitly limits
the language model to the provided context and forbids the use of external knowledge,
inference, or generalization.
If the retrieved documents do not explicitly contain the answer, the system is
instructed to return a predefined refusal response.

## The first version of Vanilla RAG and Constrained RAG
The first version of the two RAG systems was implemented using an off-the-shelf embedding model from Huggingface. For this version, the first versions of the questions were used.
The embedding model for retrieval was not trained on the labeled data, but rather downloaded.

## The second version of Vanilla RAG and Constrained RAG
The second version of the two RAG systems was implemented by training an embedding model of retrieval using the new reformulated questions. New chunks were created, and the labeling was done using 
three different Large Language Models - ChatGPT, Deepseek, and Gemini. Each chunk was assigned three labels, one from each LLM, and the decision of the final label was made based on the three labels 
proposed by the LLMs. 
Based on the gold truth, an embedding system for retrieval was trained and used on both RAG systems.

## The third version of Vanilla RAG and Constrained RAG
The third version was also based on fine-tuned embedding. This time embedding was fine-tuned using manual labeling. All chunks were reviewed by three members. The decision was made based on the
majority. Both RAG systems were implemented using this new fine-tuned system.

# Results
The three different approaches did not yield significant differences in the quality of RAG systems. Answers that were observed were similar, with the high level of hallucinations present in all three versions of the two systems, regardless of the constraint or not. Retrieval was somewhat better, as the manual labeling improved the quality of the chunks that were selected by the top-k algorithm, resulting in many more keywords retrieved in the
chunks themselves. However, the hallucinations persisted as both models kept using these buzzwords with their preexisting knowledge base to answer the questions.

# Future Works
Future work should focus on improving question design, selecting higher-quality and more focused source documents, and introducing
systematic evaluation methods for hallucination detection and answer
faithfulness, such as the RAGAS framework on different, more powerful hardware.

# Contributions by team members
Amir Saadati worked on tokenization, parsing, writing questions, and labeling the first version, as well as the follow-up version, chunking, and improvements.
Mohammadali Dehghani worked on paper retrieval, rule-based retrievers, as well as improving subsequent versions of retrieval.
Meliha Kasapovic worked on machine learning baselines, as well as the constrained RAG approach, and initial conll-u conversion.
Amina Kadic worked on baseline comparisons, Vanilla RAG implementation, and improvements on both RAG systems.
All team members worked on manual labeling interchangeably, and qualitative evaluations of different parts of the project pipeline, as well as improvements in chunking, question reformulation, and minor improvements.


---


## Environment Setup

```bash
# Create and activate a Python environment
conda create -n rag_env python=3.10 -y
conda activate rag_env

# Install all dependencies
pip install -r requirements.txt

# Download NLTK Punkt tokenizer
python - <<PY
import nltk, ssl
try:
    ssl._create_default_https_context = ssl._create_unverified_context
except:
    pass
nltk.download('punkt')
PY
