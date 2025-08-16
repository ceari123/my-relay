from flask import Flask, request, jsonify
from openai import OpenAI
import os

app = Flask(__name__)

# Read your API key from an environment variable Render will hold for you
client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])

# Put your actual vector store ID here (keep this public; the API key stays secret)
VECTOR_STORE_ID = "vs_6898e4176eac8191a1c5c1c33f12653e"
MODEL = "gpt-4.1-mini"  # you can change later to gpt-4.1 for more depth

@app.route("/", methods=["GET"])
def healthcheck():
    return jsonify({"ok": True, "service": "vector-relay"}), 200

@app.route("/vector-search", methods=["POST"])
def vector_search():
    data = request.get_json(force=True) or {}
    query = data.get("query", "").strip()
    if not query:
        return jsonify({"error": "No query provided"}), 400

    # Ask OpenAI to answer using File Search against your vector store
    resp = client.responses.create(
        model=MODEL,
        input=query,
        tools=[{"type": "file_search"}],
        tool_resources={"file_search": {"vector_store_ids": [VECTOR_STORE_ID]}},
        temperature=0
    )
    # Send back just the text for the GPT to use
    return jsonify({"answer": resp.output_text}), 200

if __name__ == "__main__":
    # IMPORTANT for Render: listen on all interfaces
    app.run(host="0.0.0.0", port=5000)
