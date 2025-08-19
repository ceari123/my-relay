# relay.py
import os
import logging
from flask import Flask, request, jsonify
from openai import OpenAI

# ---- Flask setup ----
app = Flask(__name__)
logging.basicConfig(level=logging.INFO)

ALLOWED_ORIGINS = "*"
ALLOWED_METHODS = "GET,POST,OPTIONS"
ALLOWED_HEADERS = "Content-Type, Authorization, OpenAI-Beta, X-Requested-With"

@app.after_request
def add_cors_headers(resp):
    resp.headers["Access-Control-Allow-Origin"] = ALLOWED_ORIGINS
    resp.headers["Access-Control-Allow-Methods"] = ALLOWED_METHODS
    resp.headers["Access-Control-Allow-Headers"] = ALLOWED_HEADERS
    return resp

@app.route("/status", methods=["GET", "OPTIONS"])
def status():
    if request.method == "OPTIONS":
        return ("", 204)
    return jsonify({"ok": True}), 200

@app.route("/echo", methods=["POST", "OPTIONS"])
def echo():
    if request.method == "OPTIONS":
        return ("", 204)
    body = request.get_json(silent=True) or {}
    return jsonify({"you_sent": body}), 200

# ---- OpenAI setup ----
client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))
DEFAULT_MODEL = os.environ.get("OPENAI_MODEL", "gpt-4.1-mini")

def _extract_text(resp):
    try:
        text = getattr(resp, "output_text", None)
        if text:
            return text.strip()
    except Exception:
        pass

    try:
        chunks = []
        for item in getattr(resp, "output", []) or []:
            for c in getattr(item, "content", []) or []:
                if getattr(c, "type", None) in ("output_text", "text") and getattr(c, "text", None):
                    chunks.append(c.text)
        joined = "\n".join([s for s in chunks if s]).strip()
        if joined:
            return joined
    except Exception:
        pass

    return str(resp)

@app.route("/vector-search", methods=["POST", "OPTIONS"])
def vector_search():
    if request.method == "OPTIONS":
        return ("", 204)

    body = request.get_json(silent=True) or {}
    user_input = body.get("query") or ""
    vs_id = body.get("vector_store_id") or os.environ.get("VECTOR_STORE_ID")
    model = body.get("model") or DEFAULT_MODEL

    if not user_input or not isinstance(user_input, str):
        return jsonify({"error": "Missing 'query' (string) in request body."}), 400
    if not vs_id:
        return jsonify({"error": "Missing 'vector_store_id'. Pass it in the body or set VECTOR_STORE_ID env var."}), 400

    try:
        resp = client.responses.create(
            model=model,
            tools=[{"type": "file_search"}],
            input=[
                {
                    "role": "user",
                    "content": [
                        {"type": "input_text", "text": user_input}
                    ]
                }
            ],
            file_search={"vector_store_ids": [vs_id]},
            max_output_tokens=600,
        )

        answer_text = _extract_text(resp).strip()
        if not answer_text:
            answer_text = "I couldn't find an answer in the attached notes."

        # Clean the response string to avoid GPT Builder rejection
        clean_answer = answer_text.replace("\n", " ").strip()

        return jsonify({"answer": clean_answer}), 200

    except Exception as e:
        app.logger.exception("vector-search error")
        return jsonify({"error": "Vector search failed.", "detail": str(e)}), 500

if __name__ == "__main__":
    port = int(os.environ.get("PORT", "8000"))
    app.run(host="0.0.0.0", port=port, debug=True)