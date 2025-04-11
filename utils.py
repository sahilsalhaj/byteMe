import json

def clean_query(text):
    return text.lower().strip()

def format_fund_result(fund):
    return {
        "name": fund.get("name", ""),
        "category": fund.get("category", ""),
        "fund_house": fund.get("fund_house", ""),
        "asset_class": fund.get("asset_class", ""),
        "score": round(fund.get("score", 0.0), 4)
    }

def load_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)
