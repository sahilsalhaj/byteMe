import numpy as np
import json
from embedder import get_embedding
from indexer import FundIndexer
from utils import clean_query
from fuzzywuzzy import fuzz

with open("config.json") as f:
    config = json.load(f)

indexer = FundIndexer()
indexer.build_or_load_index()

def search_fund(query):
    variants = generate_query_variants(query)
    all_results = []

    for v in variants:
        emb = get_embedding(v).astype("float32")
        D, I = indexer.index.search(np.array([emb]), config["top_k"])
        for idx in I[0]:
            all_results.append(indexer.metadata[idx])

    # Remove duplicates by fund name
    unique_results = {r["name"]: r for r in all_results}.values()

    # Rerank
    reranked = rerank_results(list(unique_results), query)

    # 👉 Return only the top result
    return reranked[:1]



def rerank_results(results, query):
    query = clean_query(query)
    query_tokens = set(query.split())

    # Heuristic: If query looks like a name (short), boost fuzzy
    is_name_query = len(query.split()) <= 3

    weights = {
        "name": 0.6,
        "category": 0.2,
        "sector": 0.1,
        "industry": 0.1
    }

    reranked = []
    for fund in results:
        score = 0.0
        print(f"--- Scoring fund: {fund.get('name')} ---")

        # Token match score
        for field, weight in weights.items():
            field_val = fund.get(field, "").lower()
            matches = sum(1 for token in query_tokens if token in field_val)
            partial_score = weight * (matches / len(query_tokens)) if query_tokens else 0.0
            score += partial_score

        # Fuzzy name boost (only if it's a name-style query)
        if is_name_query:
            fuzzy_score = fuzz.partial_ratio(query.lower(), fund["name"].lower()) / 100
            print(f"  ➤ Fuzzy score for name: {fuzzy_score}")
            score = score * 0.7 + fuzzy_score * 0.3

        fund["score"] = round(score, 4)
        reranked.append(fund)

    reranked_sorted = sorted(reranked, key=lambda x: x["score"], reverse=True)
    print("\n🏁 Final reranked results:")
    for r in reranked_sorted:
        print(f"  - {r['name']} (Score: {r['score']})")

    return reranked_sorted

from itertools import permutations

def generate_query_variants(query):
    words = query.strip().lower().split()
    variants = set()

    # Original query
    variants.add(query)

    # Add all permutations up to 3 words
    if 2 <= len(words) <= 3:
        for p in permutations(words):
            variants.add(" ".join(p))

    # Add individual words
    variants.update(words)

    # Join all words into one string (e.g. "cressendarailway")
    if len(words) > 1:
        variants.add("".join(words))

    return list(variants)

