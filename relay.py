# relay.py
import os, traceback
from flask import Flask, request, jsonify
from openai import OpenAI
import openai as openai_pkg  # only to read __version__

app = Flask(__name__)

# --- Config / Env ---
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")
VECTOR_STORE_ID = os.environ.get("VECTOR_STORE_ID")
MODEL = os.environ.get("MODEL", "gpt-4.1-mini")

if not OPENAI_API_KEY:
    raise RuntimeError("OPENAI_API_KEY not set in environment")
if not VECTOR_STORE_ID:
    raise RuntimeError("VECTOR_STORE_ID not set in environment")

client = OpenAI(api_key=OPENAI_API_KEY)

# --- Health & Status ---
@app.route("/", methods=["GET"])
def health():
    return jsonify({"ok": True, "service": "vector-relay"}), 200

@app.route("/status", methods=["GET"])
def status():
    return jsonify({
        "ok": True,
        "model": MODEL,
        "vector_store_id_present": bool(VECTOR_STORE_ID),
        "sdk_version": getattr(openai_pkg, "__version__", "unknown")
    }), 200

# --- Core handler ---
def do_vector_search():
    try:
        data = request.get_json(force=True) or {}
        q = (data.get("query") or "").strip()
        if not q:
            return jsonify({"error": "Missing 'query'"}), 400

        # Use extra_body for broader SDK compatibility (older SDKs may not accept tool_resources kwarg)
        resp = client.responses.create(
            model=MODEL,
            input=q,
            tools=[{"type": "file_search"}],
            extra_body={
                "tool_resources": {
                    "file_search": {"vector_store_ids": [VECTOR_STORE_ID]}
                }
            },
            temperature=0
        )

        # Safely extract text (covers multiple SDK output shapes)
        try:
            answer = resp.output_text
        except AttributeError:
            # Fallback for older shapes
            answer = resp.output[0].content[0].text  # may still raise if truly incompatible

        return jsonify({"answer": answer}), 200

    except Exception as e:
        # Log full traceback to Render logs and return readable error
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500

@app.route("/vector-search", methods=["POST"])
def vector_search():
    return do_vector_search()

@app.route("/vector-search/", methods=["POST"])
def vector_search_slash():
    return do_vector_search()

if __name__ == "__main__":
    # Enable debug logging so tracebacks appear in Render logs
    app.run(host="0.0.0.0", port=5000, debug=True)

