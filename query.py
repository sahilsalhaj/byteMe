import numpy as np
import json
from embedder import get_embedding
from indexer import FundIndexer

with open("config.json") as f:
    config = json.load(f)

indexer = FundIndexer()
indexer.load_data()
indexer.build_index()

def search_fund(query):
    emb = get_embedding(query).astype("float32")
    D, I = indexer.index.search(np.array([emb]), config["top_k"])
    results = [indexer.metadata[i] for i in I[0]]
    return results
