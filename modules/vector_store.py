import faiss
import numpy as np
import pickle
from pathlib import Path


class VectorStore:

    def __init__(self, db_path="vector_db"):
        self.db_path = Path(db_path)
        self.db_path.mkdir(exist_ok=True)

        self.index = None
        self.metadata = None

    def build(self, embeddings, chunks):

        dimension = embeddings.shape[1]

        self.index = faiss.IndexFlatL2(dimension)

        self.index.add(
            embeddings.astype(np.float32)
        )

        self.metadata = chunks

    def save(self):

        faiss.write_index(
            self.index,
            str(self.db_path / "faiss.index")
        )

        with open(
            self.db_path / "metadata.pkl",
            "wb"
        ) as f:

            pickle.dump(
                self.metadata,
                f
            )

    def load(self):

        self.index = faiss.read_index(
            str(self.db_path / "faiss.index")
        )

        with open(
            self.db_path / "metadata.pkl",
            "rb"
        ) as f:

            self.metadata = pickle.load(f)

    def search(self, query_embedding, k=3):

        distances, indices = self.index.search(
            query_embedding.astype(np.float32),
            k
        )

        results = []

        for idx, distance in zip(indices[0], distances[0]):

            item = self.metadata[idx].copy()

            item["score"] = float(distance)

            results.append(item)

        return results