\# Reducing LLM Hallucination in Knowledge-Intensive Tasks via Retrieval-Augmented Generation (RAG)



\## 1. Project Overview



Large Language Models (LLMs) can generate incorrect or fabricated information when answering knowledge-intensive questions. This problem is commonly known as \*\*LLM hallucination\*\*.



This project implements a Retrieval-Augmented Generation (RAG) framework to reduce hallucinations by retrieving relevant information from a document collection and providing it as contextual input to an LLM.



The system combines semantic search, document retrieval, and context-grounded response generation to improve answer accuracy and reliability.



\## 2. Objectives



\- Reduce hallucinations in LLM-generated responses.

\- Improve factual accuracy through document-based retrieval.

\- Generate answers grounded in relevant source documents.

\- Evaluate the effectiveness of RAG against a standalone LLM.

\- Analyze the impact of different retrieval Top-K values.



\## 3. Technology Stack



| Component | Technology |

|---|---|

| Programming Language | Python |

| Embedding Model | BAAI/bge-small-en-v1.5 |

| Language Model | Gemma 3 4B |

| LLM Runtime | Ollama |

| Embedding Framework | Sentence Transformers |

| Vector Storage | Custom VectorStore implementation |

| Document Processing | PDF extraction and text chunking |

| Evaluation | Python-based benchmark evaluation |



\## 4. System Architecture



The proposed system follows a Retrieval-Augmented Generation pipeline.



1\. \*\*Document Loading:\*\* Research documents are loaded from PDF files.

2\. \*\*Text Extraction:\*\* Extracted document content is converted into machine-readable text.

3\. \*\*Text Chunking:\*\* Documents are divided into smaller chunks for efficient retrieval.

4\. \*\*Embedding Generation:\*\* The BAAI/bge-small-en-v1.5 model converts text chunks into vector embeddings.

5\. \*\*Vector Storage:\*\* Embeddings and associated metadata are stored for retrieval.

6\. \*\*Semantic Retrieval:\*\* User queries are converted into embeddings and compared against stored document embeddings.

7\. \*\*Context Construction:\*\* The most relevant retrieved chunks are selected based on the Top-K setting.

8\. \*\*Response Generation:\*\* Gemma 3 4B generates responses using the retrieved context.



\## 5. Project Structure



```text

RAG-Hallucination-Reduction/

│

├── data/

│   ├── RAG-1.pdf

│   ├── RAG-2.pdf

│   └── ...

│

├── evaluation/

│   ├── evaluate.py

│   ├── evaluate\_revision.py

│   ├── questions.json

│   └── revision\_runs/

│

├── modules/

│   ├── chunker.py

│   ├── embeddings.py

│   ├── llm.py

│   ├── pdf\_loader.py

│   ├── retriever.py

│   └── vector\_store.py

│

├── vector\_db/

│   ├── faiss.index

│   └── metadata.pkl

│

├── app.py

├── query.py

├── requirements.txt

└── README.md

```



\## 6. Installation and Setup



\### Prerequisites



\- Python 3.10 or later

\- Ollama

\- Git



\### Clone the Repository



```bash

git clone https://github.com/Sathvikg143/RAG-Hallucination-Reduction.git

cd RAG-Hallucination-Reduction

```



\### Create a Virtual Environment



```bash

python -m venv venv

```



Activate the environment:



\*\*Windows:\*\*



```bash

venv\\Scripts\\activate

```



\### Install Dependencies



```bash

pip install -r requirements.txt

```



\### Set Up the Language Model



Install Ollama and download Gemma 3 4B:



```bash

ollama pull gemma3:4b

```



Ensure the Ollama service is running before executing the application.



\## 7. Experimental Evaluation



The system was evaluated using a 10-question benchmark to compare a standalone LLM against the proposed RAG approach.



Three configurations were considered:



\- Baseline Gemma 3 4B without retrieval.

\- RAG with Top-K = 5.

\- RAG with Top-K = 10.



\### Experimental Results



| Evaluation Metric | Baseline LLM | RAG (K=5) | RAG (K=10) |

|---|---:|---:|---:|

| Answer Accuracy | 60% | 80% | 100% |

| Hallucination Rate | 80% | 0% | 0% |

| Refusal Accuracy | 20% | - | 100% |



\*Note: Results are from the 10-question benchmark. They represent the observed experimental outcomes and should not be interpreted as universal performance guarantees.\*



\### Key Observations



\- The baseline LLM generated unsupported responses without external retrieval.

\- RAG with K=5 improved answer accuracy from 60% to 80%.

\- RAG with K=10 achieved 100% answer accuracy on the evaluated benchmark.

\- The RAG configurations demonstrated a 0% observed hallucination rate.

\- The results indicate that retrieving relevant contextual information can improve response reliability.



\## 8. Conclusion



This project demonstrates how Retrieval-Augmented Generation can help reduce hallucinations in knowledge-intensive tasks.



By integrating semantic retrieval with Gemma 3 4B, the system provides context-grounded answers instead of relying entirely on the language model's internal knowledge.



The experimental results show improvements in answer accuracy and factual reliability compared with the standalone LLM baseline.



Future work can explore larger benchmark datasets, different embedding models, adaptive retrieval strategies, and additional language models to further evaluate system performance.



\## 9. Future Enhancements



\- Expand the evaluation benchmark with more questions.

\- Experiment with different embedding models.

\- Optimize document chunking and retrieval strategies.

\- Explore reranking techniques for improved retrieval relevance.

\- Evaluate performance across multiple LLM architectures.



\## 10. Author



\*\*Sathvik Guptha\*\*



GitHub: \[Sathvikg143](https://github.com/Sathvikg143)



Repository: \[RAG-Hallucination-Reduction](https://github.com/Sathvikg143/RAG-Hallucination-Reduction)



