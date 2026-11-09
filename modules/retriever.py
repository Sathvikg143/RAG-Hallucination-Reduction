from modules.embeddings import EmbeddingGenerator


class Retriever:

    def __init__(self, vector_store):

        self.vector_store = vector_store

        self.embedding_model = EmbeddingGenerator()

    def retrieve(self, query, k=10):

        query_embedding = self.embedding_model.model.encode(
            [query],
            convert_to_numpy=True
        )

        results = self.vector_store.search(
            query_embedding,
            k
        )

        return results