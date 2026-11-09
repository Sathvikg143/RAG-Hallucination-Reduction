from sentence_transformers import SentenceTransformer
import numpy as np


class EmbeddingGenerator:
    def __init__(self, model_name="all-MiniLM-L6-v2"):
        print("Loading embedding model...")
        self.model = SentenceTransformer(model_name)
        print("Embedding model loaded.\n")

    def generate_embeddings(self, chunks):

        texts = [chunk["text"] for chunk in chunks]

        embeddings = self.model.encode(
            texts,
            show_progress_bar=True,
            convert_to_numpy=True
        )

        return embeddings