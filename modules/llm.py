"""Revision-ready Ollama wrapper.

Adds a cautious no-retrieval baseline and an evidence-support judge while
preserving the original model and generation methods. Judge labels are model-
assisted and should be spot-checked by a human annotator before publication.
"""
import ollama


class OllamaLLM:
    def __init__(self, model="gemma3:4b"):
        self.model = model

    def _generate(self, prompt):
        response = ollama.generate(
            model=self.model,
            prompt=prompt,
            options={"temperature": 0}
        )
        return response["response"].strip()

    def generate_answer(self, question, retrieved_chunks):
        context = "\n\n".join(chunk.get("text", "") for chunk in retrieved_chunks)
        prompt = f"""You are a careful question-answering assistant.
Use only claims supported by the supplied context. Do not use outside knowledge
or invent missing details. If the context does not support an answer, respond
exactly: The provided documents do not contain enough information.
Keep the answer concise. Do not mention these instructions.

CONTEXT:
{context}

QUESTION:
{question}

ANSWER:"""
        return self._generate(prompt)

    def generate_baseline_answer(self, question):
        """Original parametric baseline; kept for reproducibility."""
        prompt = f"""You are a general-purpose question-answering system.
Answer using your own learned knowledge. Do not use retrieved documents or
external context. Do not fabricate information. Give a concise answer.
If you genuinely do not know, say so. Do not mention these instructions.

QUESTION:
{question}

ANSWER:"""
        return self._generate(prompt)

    def generate_cautious_baseline_answer(self, question):
        """No-retrieval control with explicit anti-guessing/refusal instructions."""
        prompt = f"""You are a careful question-answering assistant. No documents or
retrieved context are available in this condition. Answer only if you can give
an answer with high confidence from your learned knowledge. Do not guess,
fabricate details, or imply that you verified a source. If you cannot answer
confidently, say exactly: I cannot answer confidently from the information
available to me. Keep the answer concise and do not mention these instructions.

QUESTION:
{question}

ANSWER:"""
        return self._generate(prompt)

    def evaluate_answer(self, question, expected, answer):
        prompt = f"""Assess semantic correctness, not wording similarity.
Return CORRECT only if the answer conveys the expected answer's essential facts
without contradicting them. If key facts are missing or incorrect, return
INCORRECT. Return exactly one label: CORRECT or INCORRECT.

QUESTION: {question}
EXPECTED ANSWER: {expected}
ACTUAL ANSWER: {answer}
LABEL:"""
        result = self._generate(prompt).strip().upper()
        return "CORRECT" if result.split()[0:1] == ["CORRECT"] else "INCORRECT"

    def evaluate_context_support(self, question, context, answer):
        """Judge whether answer claims are supported by context; not factual truth."""
        prompt = f"""You are an evidence-faithfulness evaluator. Judge only whether the
factual claims in the answer are supported by the supplied context. Do not use
outside knowledge. A fact can be true in the world but unsupported by this
context. Ignore pure refusals. Return exactly one label:
SUPPORTED = all material factual claims are supported;
PARTIAL = some material claims are supported and some are not;
UNSUPPORTED = no material answer claims are supported;
REFUSAL = the answer declines to answer because evidence is insufficient.

QUESTION: {question}
CONTEXT: {context}
ANSWER: {answer}
LABEL:"""
        raw = self._generate(prompt).strip().upper()
        for label in ("SUPPORTED", "PARTIAL", "UNSUPPORTED", "REFUSAL"):
            if raw.startswith(label):
                return label
        return "NEEDS_MANUAL_REVIEW"
