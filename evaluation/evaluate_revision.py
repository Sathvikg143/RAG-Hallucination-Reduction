"""Re-run controlled baseline/RAG evaluations and log latency/context evidence.

Run from project root after placing this file in evaluation/:
  python evaluation/evaluate_revision.py --ks 3 5 10 15

Requires the local Ollama service/model and the prebuilt vector_db. It does not
claim to calculate RAGAS. Evidence support is an LLM-assisted proxy and needs
manual validation. Precision@k/Recall@k are omitted unless relevant_chunk_ids are
provided in each question record.
"""
import argparse, json, os, sys, time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from modules.vector_store import VectorStore
from modules.retriever import Retriever
from modules.llm import OllamaLLM

REFUSAL_PHRASES = (
    "do not contain enough information", "not enough information",
    "insufficient information", "cannot answer", "can't answer",
    "unable to answer", "cannot determine", "i do not know", "i don't know",
    "cannot answer confidently", "not available in the provided context",
    "not found in the provided documents", "unable to provide"
)

def is_refusal(text):
    t = (text or "").strip().lower()
    return any(p in t for p in REFUSAL_PHRASES)

def safe_judge(llm, question, expected, answer):
    try:
        return llm.evaluate_answer(question, expected, answer)
    except Exception as exc:
        return "JUDGE_ERROR: " + str(exc)

def metrics(rows):
    ans = [r for r in rows if r["type"] == "answerable"]
    unans = [r for r in rows if r["type"] == "unanswerable"]
    correct = sum(r.get("correctness") == "CORRECT" for r in ans)
    refusals = sum(r.get("is_refusal") for r in unans)
    nonrefusal = len(unans) - refusals
    support_rows = [r for r in rows if r.get("context_support") in {"SUPPORTED", "PARTIAL", "UNSUPPORTED", "REFUSAL"}]
    supported = sum(r.get("context_support") == "SUPPORTED" for r in support_rows)
    return {
        "total_questions": len(rows),
        "answerable_questions": len(ans),
        "unanswerable_questions": len(unans),
        "answer_accuracy_pct": round(100 * correct / len(ans), 2) if ans else None,
        "refusal_accuracy_pct": round(100 * refusals / len(unans), 2) if unans else None,
        # Label deliberately avoids claiming factual falsity from absence of evidence.
        "non_refusal_rate_on_evidence_unanswerable_pct": round(100 * nonrefusal / len(unans), 2) if unans else None,
        "context_supported_response_pct_among_judged": round(100 * supported / len(support_rows), 2) if support_rows else None,
        "context_support_judged_n": len(support_rows),
        "mean_retrieval_seconds": round(sum(r.get("retrieval_seconds", 0) for r in rows) / len(rows), 4) if rows else None,
        "mean_generation_seconds": round(sum(r.get("generation_seconds", 0) for r in rows) / len(rows), 4) if rows else None,
        "mean_context_characters": round(sum(r.get("context_characters", 0) for r in rows) / len(rows), 1) if rows else None,
    }

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ks", nargs="+", type=int, default=[3, 5, 10, 15])
    parser.add_argument("--questions", default=str(ROOT / "evaluation" / "questions.json"))
    parser.add_argument("--outdir", default=str(ROOT / "evaluation" / "revision_runs"))
    parser.add_argument("--skip-original-baseline", action="store_true")
    args = parser.parse_args()
    outdir = Path(args.outdir); outdir.mkdir(parents=True, exist_ok=True)
    questions = json.loads(Path(args.questions).read_text(encoding="utf-8"))
    store = VectorStore(); store.load()
    retriever = Retriever(store); llm = OllamaLLM()
    configs = []
    if not args.skip_original_baseline:
        configs.append(("baseline_original", None, "original"))
    configs.append(("baseline_cautious", None, "cautious"))
    configs.extend((f"rag_k{k}", k, "rag") for k in args.ks)

    for name, k, mode in configs:
        print(f"\n=== {name} ===", flush=True)
        rows = []
        for item in questions:
            q = item["question"]; qtype = item["type"]; expected = item.get("expected", "")
            retrieved = []; context = ""; ret_s = 0.0
            if mode == "rag":
                t0 = time.perf_counter(); retrieved = retriever.retrieve(q, k=k); ret_s = time.perf_counter() - t0
                context = "\n\n".join(x.get("text", "") for x in retrieved)
            t0 = time.perf_counter()
            if mode == "original": answer = llm.generate_baseline_answer(q)
            elif mode == "cautious": answer = llm.generate_cautious_baseline_answer(q)
            else: answer = llm.generate_answer(q, retrieved)
            gen_s = time.perf_counter() - t0
            refusal = is_refusal(answer)
            correctness = None
            if qtype == "answerable" and not refusal:
                correctness = safe_judge(llm, q, expected, answer)
            support = None
            if mode == "rag":
                try: support = llm.evaluate_context_support(q, context, answer)
                except Exception as exc: support = "JUDGE_ERROR: " + str(exc)
            rows.append({
                "id": item.get("id"), "question": q, "type": qtype,
                "expected": expected, "answer": answer, "is_refusal": refusal,
                "correctness": correctness, "context_support": support,
                "retrieval_k": k, "retrieval_seconds": round(ret_s, 6),
                "generation_seconds": round(gen_s, 6),
                "context_characters": len(context),
                "retrieved_chunks": [{"chunk_id": c.get("chunk_id"), "source": c.get("source"), "page": c.get("page"), "score": c.get("score"), "text": c.get("text", "")} for c in retrieved]
            })
            print(f"Q{item.get('id')}: refusal={refusal}, correctness={correctness}, support={support}", flush=True)
        report = {"configuration": name, "model": llm.model, "retrieval_k": k, "prompt_condition": mode, "metrics": metrics(rows), "results": rows,
                  "note": "LLM-assisted correctness/support judgements require human spot-checking; non-refusal on an unanswerable item is not automatically proof of a factually false hallucination."}
        path = outdir / f"{name}.json"; path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
        print(json.dumps(report["metrics"], indent=2)); print(f"Saved: {path}")

if __name__ == "__main__": main()
