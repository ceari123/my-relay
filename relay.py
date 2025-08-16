from flask import Flask, request, jsonify
from openai import OpenAI
import os, traceback

app = Flask(__name__)
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")
VECTOR_STORE_ID = os.environ.get("VECTOR_STORE_ID")
MODEL = os.environ.get("MODEL", "gpt-4.1-mini")
if not OPENAI_API_KEY: raise RuntimeError("OPENAI_API_KEY not set in environment")
if not VECTOR_STORE_ID: raise RuntimeError("VECTOR_STORE_ID not set in environment")
client = OpenAI(api_key=OPENAI_API_KEY)

@app.route("/", methods=["GET"])
def health():
    return jsonify({"ok": True, "service": "vector-relay", "model": MODEL}), 200

def _handle_vector_search():
    try:
        data = request.get_json(force=True) or {}
        q = (data.get("query") or "").strip()
        if not q: return jsonify({"error":"Missing 'query'"}), 400
        resp = client.responses.create(
            model=MODEL,
            input=q,
            tools=[{"type":"file_search"}],
            tool_resources={"file_search":{"vector_store_ids":[VECTOR_STORE_ID]}},
            temperature=0
        )
        return jsonify({"answer": resp.output_text}), 200
    except Exception as e:
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500

@app.route("/vector-search", methods=["POST"])
def vector_search(): return _handle_vector_search()

@app.route("/vector-search/", methods=["POST"])
def vector_search_slash(): return _handle_vector_search()

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)

