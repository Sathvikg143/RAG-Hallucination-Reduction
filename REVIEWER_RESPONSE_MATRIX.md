# Reviewer response matrix — working draft

**Status:** This is a preparation document. Do not submit until the new experiments are run and the placeholders are replaced with actual section/page numbers and measured results.

| Reviewer comment | Required revision | Evidence needed before marking complete |
|---|---|---|
| R1: benchmark only 10 questions | Expand to 100–200 multi-domain, evidence-annotated questions if feasible; preserve 10-question set as pilot | Final benchmark file, annotation protocol, category counts |
| R1/R2: general-knowledge unanswerable questions | Replace/augment with corpus-dependent questions whose supporting evidence is genuinely absent or ambiguous | Document-search audit and manual annotation for each item |
| R1/R2: no standard RAG metrics | Add RAGAS-style faithfulness/answer relevance/context relevance; report exact package/configuration; add retrieval metrics only with relevant-passage labels | Raw outputs, evaluator settings, results, manual spot-check |
| R2: no novel algorithm | State contribution as an implementation and controlled empirical evaluation, not a new algorithm | Revised contribution statement |
| R2: baseline prompt confound | Include original baseline, cautious no-retrieval baseline, and RAG; keep decoding settings fixed | Prompt text, logs, same benchmark and model version |
| R2: weak hallucination definition | Separate correctness, context support/faithfulness, non-refusal on evidence-missing queries, and appropriate refusal | Annotation rubric and per-question labels |
| R2: insufficient reproducibility | Report PDF corpus, chunk size 700, overlap 150, all-MiniLM-L6-v2, FAISS IndexFlatL2/L2 distance, Gemma 3 4B/Ollama, temperature 0, prompts | Configuration and run logs |
| R2: only k=5,10 | Run k=3,5,10,15 (or explain limits) and report accuracy, evidence support, context size and latency | `revision_runs/*.json` files |
| R2: no retrieval/efficiency metrics | Record retrieval latency and generation latency; calculate Precision@k/Recall@k/MRR only with relevance annotations | Per-question timing and relevant-chunk IDs |
| R2: poor screenshots | Replace with formatted query/context/answer/judgment examples and readable confusion matrix | Revised figures |
| R2: conclusions overstate evidence | Limit conclusions to actual sample, uncertainty, and measured categories | Revised abstract, discussion, conclusion |

## Important implementation corrections already identified
- The supplied code uses `all-MiniLM-L6-v2`, not BAAI/bge-small-en-v1.5.
- The supplied code uses FAISS `IndexFlatL2`, not a non-FAISS vector store.
- The supplied PDF loader reads `*.pdf`; do not claim TXT support unless implemented and tested.
- The supplied chunker uses 700-character chunks with 150-character overlap.
- The supplied generation calls set temperature to 0.
- The supplied correctness judge uses the same Gemma model; treat it as model-assisted and manually validate a sample or all items.
