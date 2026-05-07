import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

class SemanticMemory:
    def __init__(self):
        self.model = SentenceTransformer("all-MiniLM-L6-v2")
        self.dimension = 384  # embedding size for MiniLM
        self.index = faiss.IndexFlatIP(self.dimension)  # cosine similarity
        self.text_chunks = []

    def add_chunks(self, chunks):
        if not chunks:
            return

        embeddings = self.model.encode(
            chunks,
            convert_to_numpy=True,
            normalize_embeddings=True
        )

        self.index.add(embeddings)
        self.text_chunks.extend(chunks)

    def search(self, query, top_k=5):
        if self.index.ntotal == 0:
            return []

        query_vec = self.model.encode(
            [query],
            convert_to_numpy=True,
            normalize_embeddings=True
        )

        scores, indices = self.index.search(query_vec, top_k)

        results = []
        for idx in indices[0]:
            if idx < len(self.text_chunks):
                results.append(self.text_chunks[idx])

        return results