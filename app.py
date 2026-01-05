from flask import Flask, request, jsonify
import subprocess

app = Flask(__name__)


@app.route("/query", methods=["POST"])
def query_model():
    user_input = request.json.get("input")
    if not user_input:
        return jsonify({"error": "No input provided"}), 400

    # Call ollama CLI with user_input
    # You may want to adjust this to keep the process alive for performance
    result = subprocess.run(
        ["ollama", "run", "gemma3:4b"], input=user_input.encode(), capture_output=True
    )
    response_text = result.stdout.decode()

    return jsonify({"response": response_text})


if __name__ == "__main__":
    app.run(
        port=11434,
    )
