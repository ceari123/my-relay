# relay.py
import os
import logging
from flask import Flask, request, jsonify
from openai import OpenAI

# ---- Flask setup ----
app = Flask(__name__)
logging.basicConfig(level=logging.INFO)

ALLOWED_ORIGINS = "*"  # lock to "https://chat.openai.com" if you prefer
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
# Ensure OPENAI_API_KEY is set in Render env vars
client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))

DEFAULT_MODEL = os.environ.get("OPENAI_MODEL", "gpt-4.1-mini")

def _extract_text(resp):
    """
    Robustly extract text from Responses API payloads.
    Prefer resp.output_text if available; otherwise, walk the content tree.
    """
    try:
        # Newer SDKs expose a convenience property
        text = getattr(resp, "output_text", None)
        if text:
            return text.strip()
    except Exception:
        pass

    # Fallback: concatenate any text parts found
    try:
        chunks = []
        for item in getattr(resp, "output", []) or []:
            for c in getattr(item, "content", []) or []:
                if getattr(c, "type", None) == "output_text" and getattr(c, "text", None):
                    chunks.append(c.text)
                # Some SDKs label it simply "text"
                if getattr(c, "type", None) == "text" and getattr(c, "text", None):
                    chunks.append(c.text)
        joined = "\n".join([s for s in chunks if s]).strip()
        if joined:
            return joined
    except Exception:
        pass

    # Last resort: string-ify
    return str(resp)

@app.route("/vector-search", methods=["POST", "OPTIONS"])
def vector_search():
    # Fast CORS preflight
    if request.method == "OPTIONS":
        return ("", 204)

    body = request.get_json(silent=True) or {}
    user_input = body.get("input") or body.get("question") or ""
    vs_id = body.get("vector_store_id") or os.environ.get("VECTOR_STORE_ID")
    model = body.get("model") or DEFAULT_MODEL

    # Basic validation
    if not user_input or not isinstance(user_input, str):
        return jsonify({"error": "Missing 'input' (string) in request body."}), 400
    if not vs_id:
        return jsonify({"error": "Missing 'vector_store_id' (pass in body or set VECTOR_STORE_ID env var)."}), 400

    try:
        # Responses API with file_search tool + message-level attachment of the vector store
        # IMPORTANT: no deprecated extra_body/tool_resources usage.
        resp = client.responses.create(
            model=model,
            tools=[{"type": "file_search"}],
            input=[
                {
                    "role": "user",
                    "content": [
                        {"type": "input_text", "text": user_input}
                    ],
                    "attachments": [
                        {"file_search": {"vector_store_ids": [vs_id]}}
                    ],
                }
            ],
            # keep outputs quick for Actions; adjust if you need longer answers
            max_output_tokens=600,
        )

        answer_text = _extract_text(resp).strip()
        if not answer_text:
            answer_text = "I couldn't find an answer in the attached notes."

        return jsonify({"answer": answer_text}), 200

    except Exception as e:
        app.logger.exception("vector-search error")
        # Do NOT leak stack traces to the client
        return jsonify({"error": "Vector search failed.", "detail": str(e)}), 500

# Local dev convenience
if __name__ == "__main__":
    port = int(os.environ.get("PORT", "8000"))
    # For local tests only; Render runs via gunicorn
    app.run(host="0.0.0.0", port=port, debug=True)