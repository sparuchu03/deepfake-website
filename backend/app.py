import os
import numpy as np
import tensorflow as tf
from flask import Flask, render_template, request, jsonify
from tensorflow.keras.preprocessing import image

# -------------------- CONFIG --------------------
MODEL_PATH = "models/deepfake_model.keras"   # <-- your .keras model
IMG_SIZE = (299, 299)

# -------------------- FLASK APP --------------------
app = Flask(__name__)

# -------------------- LOAD MODEL --------------------
print("🔄 Loading model...")
model = tf.keras.models.load_model(MODEL_PATH)
print("✅ Model loaded successfully")

# -------------------- IMAGE PREPROCESS --------------------
def preprocess_image(img_path):
    img = image.load_img(img_path, target_size=IMG_SIZE)
    img_array = image.img_to_array(img)
    img_array = img_array / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    return img_array

# -------------------- ROUTES --------------------
@app.route("/")
def index():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    if "image" not in request.files:
        return jsonify({"error": "No image uploaded"})

    file = request.files["image"]

    if file.filename == "":
        return jsonify({"error": "Empty file"})

    # Save temporarily
    temp_path = "temp.jpg"
    file.save(temp_path)

    try:
        img = preprocess_image(temp_path)
        prediction = model.predict(img)[0][0]

        confidence = round(float(prediction if prediction > 0.5 else 1 - prediction) * 100, 2)
        label = "Fake" if prediction > 0.5 else "Real"

        return jsonify({
            "label": label,
            "confidence": confidence
        })

    except Exception as e:
        return jsonify({"error": str(e)})

    finally:
        if os.path.exists(temp_path):
            os.remove(temp_path)

# -------------------- RUN SERVER --------------------
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=10000)
