from modules.vector_store import VectorStore
from modules.retriever import Retriever
from modules.llm import OllamaLLM

vector_db = VectorStore()
vector_db.load()

retriever = Retriever(vector_db)
llm = OllamaLLM()

query = input("\nAsk your question: ")

results = retriever.retrieve(query, k=10)

print("\nTop Retrieved Chunks\n")

for i, result in enumerate(results, start=1):

    print(f"Rank : {i}")
    print(f"Source : {result['source']}")
    print(f"Page : {result['page']}")
    print(f"Distance : {result['score']:.4f}\n")
    print(result["text"][:500])
    print()

answer = llm.generate_answer(query, results)

print("FINAL ANSWER")
print(answer)