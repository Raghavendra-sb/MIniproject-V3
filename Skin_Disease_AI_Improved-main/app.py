"""
app.py  (Updated)
-----------------
Flask app for Skin Disease AI.
CNN prediction is now enriched by the LangGraph + RAG pipeline.

Flow:
    User uploads image
         ↓
    CNN (MobileNetV2) predicts disease label
         ↓
    LangGraph pipeline (rag_graph.py):
        → confidence check
        → RAG retrieval from Qdrant
        → Gemini LLM explanation
        → response validation
         ↓
    Flask renders result.html with rich AI explanation
"""

from flask import Flask, request, jsonify, render_template, redirect, url_for, session, flash
from keras.models import load_model
from keras.preprocessing.image import img_to_array, load_img
import numpy as np
import os

# ── Import the LangGraph pipeline ──
from rag_graph import run_skin_disease_graph

app = Flask(__name__, static_folder="static", template_folder="templates")
app.secret_key = "your_secret_key"  # change in production

# ── Load CNN model once at startup ──
MODEL_PATH = "skin_major_model.keras"
model = load_model(MODEL_PATH)


# ─────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────
def preprocess_image(image_path, target_size=(224, 224)):
    image = load_img(image_path, target_size=target_size)
    image = img_to_array(image)
    image = np.expand_dims(image, axis=0)
    image = image / 255.0
    return image


# class index → label (unchanged from your original)
LABELS = {
    0: "Acne",
    1: "Actinic_Keratosis",
    2: "Benign_tumors",
    3: "Eczema",
    4: "Lupus",
    5: "SkinCancer",
    6: "Vasculitis",
    7: "Warts"
}


# ─────────────────────────────────────────────
# Routes
# ─────────────────────────────────────────────
@app.route("/")
def index():
    if "username" not in session:
        return redirect(url_for("login"))
    return render_template("index.html")


@app.route("/login", methods=["GET", "POST"])
def login():
    if request.method == "POST":
        username = request.form.get("username", "")
        password = request.form.get("password", "")
        if username == "admin" and password == "admin":
            session["username"] = username
            return redirect(url_for("index"))
        flash("Invalid credentials. Try admin / admin.", "error")
        return redirect(url_for("login"))
    return render_template("login.html")


@app.route("/logout")
def logout():
    session.pop("username", None)
    return redirect(url_for("login"))


@app.route("/predict", methods=["POST"])
def predict():
    if "username" not in session:
        return redirect(url_for("login"))

    if "file" not in request.files:
        flash("No file provided.", "error")
        return redirect(url_for("index"))

    file = request.files["file"]
    if file.filename == "":
        flash("No file selected.", "error")
        return redirect(url_for("index"))

    # ── Save uploaded image ──
    upload_path = os.path.join("static", "uploads")
    os.makedirs(upload_path, exist_ok=True)
    saved_path = os.path.join(upload_path, "temp.jpg")
    file.save(saved_path)

    # ── CNN Prediction ──
    image = preprocess_image(saved_path, target_size=(224, 224))
    preds = model.predict(image)
    predicted_idx = int(np.argmax(preds, axis=1)[0])
    predicted_label = LABELS.get(predicted_idx, "Unknown")
    confidence_score = float(np.max(preds))

    # ── Print all probabilities to terminal (kept from your original) ──
    print("\n===== CNN Prediction Results =====")
    probabilities = preds[0] * 100
    sorted_results = sorted(
        [(LABELS[i], prob) for i, prob in enumerate(probabilities)],
        key=lambda x: x[1],
        reverse=True
    )
    for disease, prob in sorted_results:
        print(f"  {disease}: {prob:.2f}%")
    print(f"  Predicted class: {predicted_label}")
    print(f"  Confidence: {confidence_score * 100:.4f}%")
    print("==================================\n")

    # ── LangGraph + RAG Pipeline ──
    print("🚀 Handing off to LangGraph pipeline...")
    graph_result = run_skin_disease_graph(
        disease_label=predicted_label,
        confidence_score=confidence_score
    )

    # ── Extract results from graph ──
    llm_explanation = graph_result.get("llm_explanation", "No explanation available.")
    warning = graph_result.get("warning")  # "low_confidence" or None

    # ── Render result ──
    return render_template(
        "result.html",
        image_url=url_for("static", filename="uploads/temp.jpg"),
        label=predicted_label,
        score=confidence_score,
        llm_explanation=llm_explanation,
        warning=warning,
    )


if __name__ == "__main__":
    app.run(debug=True)