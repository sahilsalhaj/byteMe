import faiss
import numpy as np
import json
import os
from embedder import get_embedding
from utils import load_json

class FundIndexer:
    def __init__(self):
        self.index_path = "faiss.index"
        self.metadata_path = "metadata.json"
        self.index = None
        self.metadata = []
        self.funds = []

    def load_data(self, path="data/cleaned_stock_data.json"):
        self.funds = load_json(path)
        print(f"📦 Loaded {len(self.funds)} records from {path}")

    def build_index(self):
        embeddings = []
        self.metadata = []

        for fund in self.funds:
            query_string = fund["name"] + " " + fund["category"] + " " + fund["sector"] + " " + fund["industry"]
            emb = get_embedding(query_string)
            embeddings.append(emb)
            self.metadata.append(fund)

        embeddings = np.array(embeddings).astype("float32")
        self.index = faiss.IndexFlatL2(embeddings.shape[1])
        self.index.add(embeddings)

    def build_or_load_index(self):
        if os.path.exists(self.index_path) and os.path.exists(self.metadata_path):
            print("✅ FAISS index and metadata found. Loading...")
            self.index = faiss.read_index(self.index_path)
            with open(self.metadata_path, "r", encoding="utf-8") as f:
                self.metadata = json.load(f)
        else:
            print("🚧 FAISS index not found. Building new index...")
            self.load_data()
            self.build_index()
            faiss.write_index(self.index, self.index_path)
            with open(self.metadata_path, "w", encoding="utf-8") as f:
                json.dump(self.metadata, f)
