from pathlib import Path
import pickle
import time

import numpy as np
import tensorflow as tf
from flask import Flask, jsonify, render_template, request
from tensorflow.keras.preprocessing.sequence import pad_sequences


BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "next_word.h5"
TOKENIZER_PATH = BASE_DIR / "tokenizer.pkl"

app = Flask(__name__)


def load_artifacts():
    if not MODEL_PATH.is_file():
        raise FileNotFoundError("The trained model file is missing.")
    if not TOKENIZER_PATH.is_file():
        raise FileNotFoundError("The tokenizer file is missing.")

    model = tf.keras.models.load_model(MODEL_PATH, compile=False)
    with TOKENIZER_PATH.open("rb") as tokenizer_file:
        tokenizer = pickle.load(tokenizer_file)

    input_shape = model.input_shape
    if len(input_shape) != 2 or input_shape[1] is None:
        raise ValueError("The loaded model does not expose a fixed sequence length.")

    return model, tokenizer, int(input_shape[1])


MODEL, TOKENIZER, MAX_SEQUENCE_LENGTH = load_artifacts()


def predict_word(tokens):
    padded = pad_sequences(
        [tokens],
        maxlen=MAX_SEQUENCE_LENGTH,
        padding="pre",
        truncating="pre",
    )
    probabilities = MODEL.predict(padded, verbose=0)
    predicted_index = int(np.argmax(probabilities[0]))
    return TOKENIZER.index_word.get(predicted_index)


def generate_words(text, count):
    input_tokens = TOKENIZER.texts_to_sequences([text])[0]
    if not input_tokens:
        raise ValueError("Enter at least one known or unknown word to begin prediction.")

    generated = []
    context_tokens = input_tokens[:]
    for _ in range(count):
        next_word = predict_word(context_tokens)
        if not next_word:
            raise RuntimeError("The model returned an index without a matching tokenizer word.")
        generated.append(next_word)
        context_tokens.append(TOKENIZER.word_index.get(next_word, 0))

    return generated


@app.get("/")
def index():
    return render_template("index.html")


@app.get("/api/health")
def health():
    return jsonify(
        {
            "status": "ok",
            "model_loaded": MODEL is not None,
            "tokenizer_loaded": TOKENIZER is not None,
            "sequence_length": MAX_SEQUENCE_LENGTH,
        }
    )


@app.post("/api/predict")
def predict():
    payload = request.get_json(silent=True)
    if not isinstance(payload, dict):
        return jsonify({"error": "Request body must be a JSON object."}), 400

    text = payload.get("text")
    num_words = payload.get("num_words", 1)
    if not isinstance(text, str) or not text.strip():
        return jsonify({"error": "Please provide non-empty text."}), 400
    if isinstance(num_words, bool) or not isinstance(num_words, int) or not 1 <= num_words <= 10:
        return jsonify({"error": "num_words must be an integer from 1 to 10."}), 400

    started = time.perf_counter()
    try:
        generated = generate_words(text.strip(), num_words)
    except (ValueError, RuntimeError) as error:
        return jsonify({"error": str(error)}), 422
    except tf.errors.OpError:
        app.logger.exception("Model inference failed")
        return jsonify({"error": "Prediction could not be completed."}), 500

    continuation = " ".join(generated)
    return jsonify(
        {
            "original_input": text.strip(),
            "predicted_continuation": continuation,
            "generated_sentence": f"{text.strip()} {continuation}",
            "generated_words": generated,
            "number_of_words": len(generated),
            "execution_time_ms": round((time.perf_counter() - started) * 1000, 2),
        }
    )


if __name__ == "__main__":
    app.run(debug=False)
