# relay.py — unified relay with vector store search (fixed)
import os, sys, traceback
from flask import Flask, request, jsonify
from openai import OpenAI
import openai as openai_pkg  # for version reporting

app = Flask(__name__)

# --- Config via environment ---
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
VECTOR_STORE_ID = os.getenv("VECTOR_STORE_ID")  # set this in Render → Environment
MODEL = os.getenv("MODEL", "gpt-4o-mini")  # recommend a 4o variant for file_search

if not OPENAI_API_KEY:
    print("[BOOT] ERROR: OPENAI_API_KEY not set", file=sys.stderr, flush=True)
if not VECTOR_STORE_ID:
    print("[BOOT] ERROR: VECTOR_STORE_ID not set", file=sys.stderr, flush=True)

client = OpenAI(api_key=OPENAI_API_KEY) if OPENAI_API_KEY else None

# --- Simple request logging ---
@app.before_request
def _in():
    try: print(f"[REQ] {request.method} {request.path}", flush=True)
    except: pass

@app.after_request
def _out(resp):
    try: print(f"[RESP] {request.method} {request.path} -> {resp.status_code}", flush=True)
    except: pass
    return resp

# --- Health / status ---
@app.route("/", methods=["GET"])
def health():
    return jsonify({"ok": True, "service": "vector-relay"}), 200

@app.route("/status", methods=["GET"])
def status():
    return jsonify({
        "ok": True,
        "model": MODEL,
        "sdk_version": getattr(openai_pkg, "__version__", "unknown"),
        "vector_store_id_present": bool(VECTOR_STORE_ID)
    }), 200

# --- Echo (no OpenAI, just to test POST+JSON) ---
@app.route("/echo", methods=["POST"])
def echo():
    try:
        data = request.get_json(force=True) or {}
        return jsonify({"received": data}), 200
    except Exception as e:
        traceback.print_exc()
        return jsonify({"error": str(e)}), 400

# --- Vector search via Responses API + file_search tool ---
def _do_vector_search():
    try:
        data = request.get_json(force=True) or {}
        q = (data.get("query") or "").strip()
        top_k = int(data.get("top_k") or 6)
        if not q:
            return jsonify({"error": "Missing 'query'"}), 400

        if not client:
            return jsonify({"error": "OPENAI_API_KEY not set on server"}), 500
        if not VECTOR_STORE_ID:
            return jsonify({"error": "VECTOR_STORE_ID not set on server"}), 500

        # ✅ Current, supported payload (no extra_body/tool_resources)
        resp = client.responses.create(
            model=MODEL,
            input=q,
            tools=[{
                "type": "file_search",
                "vector_store_ids": [VECTOR_STORE_ID],
                "max_num_results": top_k
            }],
            tool_choice="required",   # force using file_search instead of guessing
            temperature=0
        )

        # Robust output extraction across SDK shapes
        answer = getattr(resp, "output_text", None)
        if not answer:
            # Fallback: walk the structured output
            try:
                answer = resp.output[0].content[0].text
            except Exception:
                answer = None

        if not answer:
            return jsonify({"error": "No answer produced"}), 500

        return jsonify({"answer": answer}), 200

    except Exception as e:
        traceback.print_exc()
        return jsonify({"error": f"vector_search_failed: {e}"}), 500

@app.route("/vector-search", methods=["POST"])
def vector_search():
    return _do_vector_search()

@app.route("/vector-search/", methods=["POST"])
def vector_search_slash():
    return _do_vector_search()

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
