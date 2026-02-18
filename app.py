import os
from functools import wraps
from io import BytesIO

import numpy as np
from PIL import Image
from dotenv import load_dotenv
from flask import Flask, request, jsonify, render_template
from werkzeug.utils import secure_filename
import tensorflow as tf
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input

# ------------------------------
# Load environment & config
# ------------------------------
load_dotenv()  # reads .env in project root

MODEL_PATH = os.getenv("MODEL_PATH", "ai_image_detector.h5")
API_KEYS_ENV = os.getenv("API_KEYS", "")
UPLOAD_FOLDER = os.getenv("UPLOAD_FOLDER", "static/uploads")
DEBUG_MODE = os.getenv("DEBUG", "True").lower() in ("1", "true", "yes")

# parse allowed API keys (comma separated)
ALLOWED_KEYS = set(k.strip() for k in API_KEYS_ENV.split(",") if k.strip())

os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# ------------------------------
# Load model
# ------------------------------
if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(f"Model file not found at '{MODEL_PATH}'. Place your .h5 model there.")

model = tf.keras.models.load_model(
    MODEL_PATH,
    compile=False
)


# ------------------------------
# Flask app
# ------------------------------
app = Flask(__name__)
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER

IMG_SIZE = (224, 224)

def prepare_image(file_storage):
    """Read uploaded FileStorage and preprocess for MobileNetV2"""
    file_storage.seek(0)
    img = Image.open(BytesIO(file_storage.read())).convert("RGB")
    img = img.resize(IMG_SIZE)
    arr = image.img_to_array(img)
    arr = np.expand_dims(arr, axis=0)
    arr = preprocess_input(arr)
    file_storage.seek(0)
    return arr

def require_api_key(f):
    @wraps(f)
    def decorated(*args, **kwargs):
        if not ALLOWED_KEYS:
            return jsonify({"error": "Server misconfiguration: no API key set"}), 500
        key = request.headers.get("x-api-key") or request.args.get("api_key")
        if not key or key not in ALLOWED_KEYS:
            return jsonify({"error": "Unauthorized - invalid API key"}), 401
        return f(*args, **kwargs)
    return decorated

# ------------------------------
# UI routes
# ------------------------------
@app.route("/")
def index():
    return render_template("index.html")

@app.route("/detect", methods=["POST"])
def detect():
    if "image" not in request.files:
        return render_template("index.html", error="❌ No file uploaded!")
    file = request.files["image"]
    if file.filename == "":
        return render_template("index.html", error="❌ No file selected!")
    try:
        img_arr = prepare_image(file)
        pred = model.predict(img_arr)[0][0]
        if pred > 0.5:
            result = "✅ Real Image"
            confidence = float(pred * 100)
        else:
            result = "🤖 AI-Generated Image"
            confidence = float((1 - pred) * 100)
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config["UPLOAD_FOLDER"], filename)
        # ensure unique filename to avoid overwrite
        base, ext = os.path.splitext(filename)
        counter = 1
        while os.path.exists(filepath):
            filename = f"{base}_{counter}{ext}"
            filepath = os.path.join(app.config["UPLOAD_FOLDER"], filename)
            counter += 1
        file.save(filepath)
        return render_template("detect.html",
                               result=f"{result} ({confidence:.2f}% confidence)",
                               img_path=filepath)
    except Exception as e:
        return render_template("index.html", error=f"⚠️ Prediction error: {e}")

# ------------------------------
# API route (protected)
# ------------------------------
@app.route("/api/detect", methods=["POST"])
@require_api_key
def api_detect():
    if "image" not in request.files:
        return jsonify({"error": "No file uploaded"}), 400
    file = request.files["image"]
    if file.filename == "":
        return jsonify({"error": "No file selected"}), 400
    try:
        img_arr = prepare_image(file)
        pred = model.predict(img_arr)[0][0]
        if pred > 0.5:
            label = "real"
            confidence = float(pred)
        else:
            label = "ai"
            confidence = float(1 - pred)
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config["UPLOAD_FOLDER"], filename)
        base, ext = os.path.splitext(filename)
        counter = 1
        while os.path.exists(filepath):
            filename = f"{base}_{counter}{ext}"
            filepath = os.path.join(app.config["UPLOAD_FOLDER"], filename)
            counter += 1
        file.save(filepath)
        # public URL for returned file (works on localhost; in prod adjust)
        host = request.host_url.rstrip('/')
        file_url = f"{host}/{filepath.replace(os.path.sep, '/')}"
        return jsonify({
            "result": label,
            "confidence": round(confidence * 100, 2),
            "image_url": file_url
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500

# ------------------------------
# Run
# ------------------------------
if __name__ == "__main__":
    app.run(debug=DEBUG_MODE)
