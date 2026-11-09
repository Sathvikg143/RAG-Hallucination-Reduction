# RAG conference major-revision kit

## What the supplied code actually does
- PDF-only loader (`data/*.pdf`), not PDF/TXT ingestion.
- `RecursiveCharacterTextSplitter`: 700 characters with 150-character overlap.
- `all-MiniLM-L6-v2` embeddings.
- FAISS `IndexFlatL2` and pickle metadata (so the paper must not call this a non-FAISS custom vector store).
- Gemma 3 4B via Ollama, temperature 0.
- Existing benchmark: 10 questions (5 answerable, 5 unanswerable).
- Existing judge is the same Gemma model and is not independent; results should be described as model-assisted unless manually verified.

## Immediate code changes
1. Back up the current `modules/llm.py` and `evaluation/evaluate.py`.
2. Replace `modules/llm.py` with the supplied `modules/llm.py` in this kit.
3. Copy `evaluation/evaluate_revision.py` into your project's `evaluation/` folder.
4. Confirm Ollama is running and `gemma3:4b` is available. Then run from the project root:
   `python evaluation/evaluate_revision.py --ks 3 5 10 15`
5. Results are saved separately in `evaluation/revision_runs/`. Do not overwrite the original result files.

## Interpretation rules
- The original 10-question benchmark must remain labelled as the pilot/original evaluation.
- Questions 6, 8 (capital of France / World Cup winner) are general facts; they are not valid evidence-missing questions merely because those facts are absent from a research corpus.
- Do not label every non-refusal as a factually false hallucination. Report it as a non-refusal on an evidence-unanswerable item, then separately evaluate factuality and context support.
- The new cautious baseline has no retrieval and asks the model not to guess. Compare it with the original baseline and RAG to expose prompt effects.
- The script records latency and context length. It does NOT compute RAGAS metrics or retrieval precision/recall automatically. Those require the RAGAS setup and/or manually annotated relevant chunks.
- `benchmark_extension_template.json` is a template, not an expanded or validated benchmark. Fill it with evidence-checked questions before claiming a larger dataset.

## Paper corrections required regardless of new results
- Replace BAAI/bge-small-en-v1.5 with `all-MiniLM-L6-v2` wherever the submitted implementation is the source of truth.
- Replace “custom VectorStore does not use FAISS” with “a project-specific VectorStore wrapper uses FAISS IndexFlatL2 and stores chunk metadata in metadata.pkl.”
- Describe the source corpus as PDFs unless TXT ingestion is implemented and tested.
- Report chunk size 700 characters and overlap 150 characters.
- Report LLM temperature 0.
- Clarify that the original baseline prompt and RAG prompt differ. Use the new cautious no-retrieval control for a closer comparison.
- Do not claim 100% generalizable accuracy or that hallucinations are eliminated. State the exact dataset size and evaluation procedure.
- Do not claim RAGAS/ARES, Precision@k/Recall@k, or MRR results unless those evaluations have actually been run and documented.
