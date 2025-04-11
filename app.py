from flask import Flask, request, jsonify
from query_engine import search_fund
from utils import format_fund

app = Flask(__name__)

@app.route("/query", methods=["POST"])
def query():
    data = request.get_json()
    user_query = data.get("query")
    if not user_query:
        return jsonify({"error": "Missing 'query' field"}), 400

    matches = search_fund(user_query)
    results = [format_fund(f) for f in matches]
    return jsonify({"matches": results})

if __name__ == "__main__":
    app.run(debug=True)

