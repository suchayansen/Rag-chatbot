import faiss
from rag.embeddings import embed

class VectorStore:
    def __init__(self):
        self.index = None
        self.texts = []

    def build(self, chunks):
        self.texts = chunks
        vectors = embed(chunks).astype("float32")
        self.index = faiss.IndexFlatL2(vectors.shape[1])
        self.index.add(vectors)

    def search(self, query, k=5):
        if self.index is None:
            return []
        q_vec = embed([query]).astype("float32")
        _, idx = self.index.search(q_vec, k)
        return [self.texts[i] for i in idx[0]]
