import faiss
import numpy as np
import json
from embedder import get_embedding

class FundIndexer:
    def __init__(self):
        self.funds = []
        self.metadata = []
        self.index = None

    def load_data(self, path="data/funds.json"):
        with open(path) as f:
            self.funds = json.load(f)

    def build_index(self):
        embeddings = []
        for fund in self.funds:
            query_string = fund["name"] + " " + fund["fund_house"] + " " + fund["category"]
            emb = get_embedding(query_string)
            embeddings.append(emb)
            self.metadata.append(fund)

        embeddings = np.array(embeddings).astype("float32")
        self.index = faiss.IndexFlatL2(embeddings.shape[1])
        self.index.add(embeddings)

    def save_index(self):
        faiss.write_index(self.index, "faiss.index")

    def load_index(self):
        self.index = faiss.read_index("faiss.index")
