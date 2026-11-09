import json
import sys
import os

# Project root

PROJECT_ROOT = os.path.dirname(
    os.path.dirname(
        os.path.abspath(__file__)
    )
)

sys.path.append(PROJECT_ROOT)

# Imports

from modules.vector_store import VectorStore
from modules.retriever import Retriever
from modules.llm import OllamaLLM


# Configuration

if len(sys.argv) > 1:
    RETRIEVAL_K = int(sys.argv[1])
else:
    RETRIEVAL_K = 10


QUESTIONS_FILE = os.path.join(
    PROJECT_ROOT,
    "evaluation",
    "questions.json"
)

RAG_RESULTS_FILE = os.path.join(
    PROJECT_ROOT,
    "evaluation",
    "rag_results.json"
)

BASELINE_RESULTS_FILE = os.path.join(
    PROJECT_ROOT,
    "evaluation",
    "baseline_results.json"
)

COMPARISON_RESULTS_FILE = os.path.join(
    PROJECT_ROOT,
    "evaluation",
    "comparison_results.json"
)



# Load evaluation questions

with open(
    QUESTIONS_FILE,
    "r",
    encoding="utf-8"
) as f:

    questions = json.load(f)


# ============================================================
# Load components
# ============================================================

print("Loading vector database...")

vector_db = VectorStore()
vector_db.load()

print("Loading retriever...")

retriever = Retriever(vector_db)

print("Loading LLM...")

llm = OllamaLLM()


# ============================================================
# Refusal detection
# ============================================================

def is_refusal(answer):

    text = answer.strip().lower()

    refusal_phrases = [

        "the provided documents do not contain enough information",

        "do not contain enough information",

        "does not contain enough information",

        "not enough information",

        "insufficient information",

        "not enough context",

        "insufficient context",

        "cannot answer",

        "can't answer",

        "unable to answer",

        "cannot be answered",

        "can't be answered",

        "unanswerable",

        "information is not available",

        "information is unavailable",

        "not found in the provided documents",

        "not found in the documents",

        "cannot provide",

        "can't provide",

        "unable to provide",

        "i don't have enough information",

        "i do not have enough information",

        "i cannot determine",

        "i can't determine",

        "cannot determine",

        "i don't have access",

        "i do not have access",

        "real-time information",

        "real time information"

    ]

    for phrase in refusal_phrases:

        if phrase in text:
            return True

    return False


# ============================================================
# Evaluate answer correctness
# ============================================================

def evaluate_answer(question, expected, answer):

    return llm.evaluate_answer(
        question,
        expected,
        answer
    )


# ============================================================
# Evaluate one answer
# ============================================================

def evaluate_single_answer(
    question_type,
    question,
    expected,
    answer
):

    refused = is_refusal(answer)

    # --------------------------------------------------------
    # Unanswerable question
    # --------------------------------------------------------

    if question_type == "unanswerable":

        if refused:
            return "CORRECT_REFUSAL"

        return "HALLUCINATION"

    # --------------------------------------------------------
    # Answerable question
    # --------------------------------------------------------

    if refused:

        return "FAILED_TO_ANSWER"

    correctness = evaluate_answer(
        question,
        expected,
        answer
    )

    if correctness == "CORRECT":

        return "CORRECT"

    return "INCORRECT"


# ============================================================
# Calculate metrics
# ============================================================

def calculate_metrics(results):

    answerable = [
        r for r in results
        if r["type"] == "answerable"
    ]

    unanswerable = [
        r for r in results
        if r["type"] == "unanswerable"
    ]

    correct_refusals = [
        r for r in unanswerable
        if r["evaluation"] == "CORRECT_REFUSAL"
    ]

    hallucinations = [
        r for r in unanswerable
        if r["evaluation"] == "HALLUCINATION"
    ]

    failed_answers = [
        r for r in answerable
        if r["evaluation"] == "FAILED_TO_ANSWER"
    ]

    correct_answers = [
        r for r in answerable
        if r["evaluation"] == "CORRECT"
    ]

    incorrect_answers = [
        r for r in answerable
        if r["evaluation"] == "INCORRECT"
    ]

    # --------------------------------------------------------
    # Percentages
    # --------------------------------------------------------

    if len(unanswerable) > 0:

        hallucination_rate = (
            len(hallucinations)
            / len(unanswerable)
        ) * 100

        refusal_accuracy = (
            len(correct_refusals)
            / len(unanswerable)
        ) * 100

    else:

        hallucination_rate = 0
        refusal_accuracy = 0

    if len(answerable) > 0:

        answer_accuracy = (
            len(correct_answers)
            / len(answerable)
        ) * 100

    else:

        answer_accuracy = 0

    return {

        "total_questions": len(results),

        "answerable_questions": len(answerable),

        "unanswerable_questions": len(unanswerable),

        "correct_refusals": len(correct_refusals),

        "hallucinations": len(hallucinations),

        "correct_answers": len(correct_answers),

        "incorrect_answers": len(incorrect_answers),

        "failed_answers": len(failed_answers),

        "answer_accuracy": round(
            answer_accuracy,
            2
        ),

        "hallucination_rate": round(
            hallucination_rate,
            2
        ),

        "refusal_accuracy": round(
            refusal_accuracy,
            2
        )
    }


# ============================================================
# Print metrics
# ============================================================

def print_metrics(title, metrics):

    print("\n")
    print("=" * 70)
    print(title)
    print("=" * 70)

    print(
        f"Total Questions       : "
        f"{metrics['total_questions']}"
    )

    print(
        f"Answerable Questions  : "
        f"{metrics['answerable_questions']}"
    )

    print(
        f"Unanswerable Questions: "
        f"{metrics['unanswerable_questions']}"
    )

    print("\nHallucination Analysis")
    print("-" * 70)

    print(
        f"Correct Refusals      : "
        f"{metrics['correct_refusals']}/"
        f"{metrics['unanswerable_questions']}"
    )

    print(
        f"Hallucinations        : "
        f"{metrics['hallucinations']}/"
        f"{metrics['unanswerable_questions']}"
    )

    print(
        f"Correct Answers       : "
        f"{metrics['correct_answers']}/"
        f"{metrics['answerable_questions']}"
    )

    print(
        f"Incorrect Answers     : "
        f"{metrics['incorrect_answers']}/"
        f"{metrics['answerable_questions']}"
    )

    print(
        f"Failed Answers        : "
        f"{metrics['failed_answers']}/"
        f"{metrics['answerable_questions']}"
    )

    print("\nMetrics")
    print("-" * 70)

    print(
        f"Answer Accuracy       : "
        f"{metrics['answer_accuracy']:.2f}%"
    )

    print(
        f"Hallucination Rate    : "
        f"{metrics['hallucination_rate']:.2f}%"
    )

    print(
        f"Refusal Accuracy      : "
        f"{metrics['refusal_accuracy']:.2f}%"
    )


# ============================================================
# Run BASELINE evaluation
# ============================================================

print("\n")
print("=" * 70)
print("BASELINE LLM EVALUATION")
print("=" * 70)

print("Using Gemma 3:4B without retrieval.")
print("Retrieval is NOT used for the baseline.\n")

baseline_results = []

for item in questions:

    question_id = item["id"]
    question = item["question"]
    question_type = item["type"]
    expected = item["expected"]

    print("=" * 70)

    print(f"Question {question_id}")
    print(f"Type: {question_type}")
    print(f"Question: {question}")

    # --------------------------------------------------------
    # Direct LLM generation
    # --------------------------------------------------------

    answer = llm.generate_baseline_answer(
        question
    )

    answer = answer.strip()

    print(f"Answer: {answer}")

    # --------------------------------------------------------
    # Evaluate
    # --------------------------------------------------------

    evaluation = evaluate_single_answer(
        question_type,
        question,
        expected,
        answer
    )

    print(f"Evaluation: {evaluation}")

    # --------------------------------------------------------
    # Store
    # --------------------------------------------------------

    baseline_results.append(
        {
            "id": question_id,
            "question": question,
            "type": question_type,
            "expected": expected,
            "answer": answer,
            "evaluation": evaluation
        }
    )


# ============================================================
# Baseline metrics
# ============================================================

baseline_metrics = calculate_metrics(
    baseline_results
)

print_metrics(
    "BASELINE RESULTS",
    baseline_metrics
)


# ============================================================
# Save baseline results
# ============================================================

with open(
    BASELINE_RESULTS_FILE,
    "w",
    encoding="utf-8"
) as f:

    json.dump(
        {
            "system": "baseline",
            "model": llm.model,
            "metrics": baseline_metrics,
            "results": baseline_results
        },
        f,
        indent=4
    )

print("\nBaseline results saved to:")
print("evaluation/baseline_results.json")


# ============================================================
# Run RAG evaluation
# ============================================================

print("\n")
print("=" * 70)
print("RAG EVALUATION")
print("=" * 70)

print(f"Retrieval k: {RETRIEVAL_K}\n")

rag_results = []

for item in questions:

    question_id = item["id"]
    question = item["question"]
    question_type = item["type"]
    expected = item["expected"]

    print("=" * 70)

    print(f"Question {question_id}")
    print(f"Type: {question_type}")
    print(f"Question: {question}")

    # --------------------------------------------------------
    # Retrieve relevant chunks
    # --------------------------------------------------------

    retrieved_chunks = retriever.retrieve(
        question,
        k=RETRIEVAL_K
    )

    # --------------------------------------------------------
    # Generate RAG answer
    # --------------------------------------------------------

    answer = llm.generate_answer(
        question,
        retrieved_chunks
    )

    answer = answer.strip()

    print(f"Answer: {answer}")

    # --------------------------------------------------------
    # Evaluate
    # --------------------------------------------------------

    evaluation = evaluate_single_answer(
        question_type,
        question,
        expected,
        answer
    )

    print(f"Evaluation: {evaluation}")

    # --------------------------------------------------------
    # Store
    # --------------------------------------------------------

    rag_results.append(
        {
            "id": question_id,
            "question": question,
            "type": question_type,
            "expected": expected,
            "answer": answer,
            "evaluation": evaluation
        }
    )


# ============================================================
# RAG metrics
# ============================================================

rag_metrics = calculate_metrics(
    rag_results
)

print_metrics(
    "RAG RESULTS",
    rag_metrics
)


# ============================================================
# Save RAG results
# ============================================================

with open(
    RAG_RESULTS_FILE,
    "w",
    encoding="utf-8"
) as f:

    json.dump(
        {
            "system": "RAG",
            "model": llm.model,
            "retrieval_k": RETRIEVAL_K,
            "metrics": rag_metrics,
            "results": rag_results
        },
        f,
        indent=4
    )

print("\nRAG results saved to:")
print("evaluation/rag_results.json")


# ============================================================
# Compare Baseline vs RAG
# ============================================================

baseline_hallucination = (
    baseline_metrics["hallucination_rate"]
)

rag_hallucination = (
    rag_metrics["hallucination_rate"]
)

baseline_accuracy = (
    baseline_metrics["answer_accuracy"]
)

rag_accuracy = (
    rag_metrics["answer_accuracy"]
)

baseline_refusal = (
    baseline_metrics["refusal_accuracy"]
)

rag_refusal = (
    rag_metrics["refusal_accuracy"]
)


# ------------------------------------------------------------
# Hallucination reduction
# ------------------------------------------------------------

if baseline_hallucination > 0:

    hallucination_reduction = (
        (
            baseline_hallucination
            - rag_hallucination
        )
        / baseline_hallucination
    ) * 100

else:

    hallucination_reduction = 0


comparison = {

    "baseline": baseline_metrics,

    "rag": rag_metrics,

    "improvement": {

        "hallucination_reduction_percentage":
            round(
                hallucination_reduction,
                2
            ),

        "answer_accuracy_change":
            round(
                rag_accuracy
                - baseline_accuracy,
                2
            ),

        "refusal_accuracy_change":
            round(
                rag_refusal
                - baseline_refusal,
                2
            )
    }
}


# ============================================================
# Print comparison
# ============================================================

print("\n")
print("=" * 70)
print("BASELINE VS RAG COMPARISON")
print("=" * 70)

print(
    f"{'Metric':<30}"
    f"{'Baseline':>15}"
    f"{'RAG':>15}"
)

print("-" * 70)

print(
    f"{'Answer Accuracy':<30}"
    f"{baseline_accuracy:>14.2f}%"
    f"{rag_accuracy:>14.2f}%"
)

print(
    f"{'Hallucination Rate':<30}"
    f"{baseline_hallucination:>14.2f}%"
    f"{rag_hallucination:>14.2f}%"
)

print(
    f"{'Refusal Accuracy':<30}"
    f"{baseline_refusal:>14.2f}%"
    f"{rag_refusal:>14.2f}%"
)

print("-" * 70)

print(
    f"Hallucination Reduction: "
    f"{hallucination_reduction:.2f}%"
)


# ============================================================
# Save comparison
# ============================================================

with open(
    COMPARISON_RESULTS_FILE,
    "w",
    encoding="utf-8"
) as f:

    json.dump(
        comparison,
        f,
        indent=4
    )


print("\nComparison results saved to:")
print("evaluation/comparison_results.json")

print("\n")
print("=" * 70)
print("EXPERIMENT COMPLETED")
print("=" * 70)