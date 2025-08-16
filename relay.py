# relay.py
import os
from flask import Flask, request
from openai import OpenAI

app = Flask(__name__)
client = OpenAI()

# Health check route (GET only)
@app.route("/", methods=["GET"])
def home():
    return {"ok": True, "message": "Relay is running"}

# Vector search route (POST only)
@app.route("/vector-search", methods=["POST"])
def vector_search():
    data = request.get_json()
    query = data.get("query", "")
    if not query:
        return {"error": "Missing query"}, 400

    try:
        resp = client.responses.create(
            model="gpt-4.1-mini",
            input=query,
            tool_resources={
                "file_search": {
                    "vector_store_ids": [os.environ["VECTOR_STORE_ID"]]
                }
            },
        )
        # Extract the first text chunk
        answer = resp.output[0].content[0].text
        return {"answer": answer}
    except Exception as e:
        return {"error": str(e)}, 500

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
