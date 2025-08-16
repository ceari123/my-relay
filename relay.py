from flask import Flask, request, jsonify
from openai import OpenAI
import os

app = Flask(__name__)
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# Health check endpoint
@app.route("/", methods=["GET"])
def home():
    return "Relay server is running", 200

# Vector search endpoint
@app.route("/vector-search", methods=["POST"])
def vector_search():
    try:
        data = request.get_json()

        # Extract input
        query = data.get("query", "")
        vector_store_id = data.get("vector_store_id")

        if not query or not vector_store_id:
            return jsonify({"error": "Missing 'query' or 'vector_store_id'"}), 400

        # Call OpenAI
        response = client.chat.completions.create(
            model="gpt-4.1",
            messages=[
                {"role": "system", "content": "You are a helpful study assistant."},
                {"role": "user", "content": f"Use vector store {vector_store_id} to answer: {query}"}
            ]
        )

        # Extract text from the response
        answer = response.choices[0].message["content"]
        return jsonify({"answer": answer})

    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
