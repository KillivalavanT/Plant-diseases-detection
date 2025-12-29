# app.py (full – replace existing)
import os
import io
import base64
import datetime
import numpy as np
import requests
from PIL import Image
from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas
from flask import Flask, render_template, request, jsonify, send_file, url_for, redirect
from tensorflow.keras.preprocessing.image import load_img, img_to_array
import tensorflow as tf

# Database
from flask_sqlalchemy import SQLAlchemy

# ========== CONFIG ==========
MODEL_PATH = os.path.join("models", "leaf_mobilenetv2_best_saved")
CLASS_NAMES = ['Blight', 'Common_Rust', 'Gray_Leaf_Spot', 'Healthy']
UPLOAD_FOLDER = os.path.join("static", "uploads")
HEATMAP_FOLDER = os.path.join(UPLOAD_FOLDER, "heatmaps")
ALLOWED_EXTENSIONS = {"jpg", "jpeg", "png"}
DB_PATH = "sqlite:///predictions.db"
OPENWEATHER_API_KEY = os.environ.get("OPENWEATHER_API_KEY")  # set this in your environment
# ============================

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['SQLALCHEMY_DATABASE_URI'] = DB_PATH
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False

db = SQLAlchemy(app)

# create folders if missing
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(HEATMAP_FOLDER, exist_ok=True)

# global model + img size
model = None
IMG_SIZE = None


# ========== DB MODELS ==========
class Prediction(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    filename = db.Column(db.String(256))
    predicted_class = db.Column(db.String(64))
    confidence = db.Column(db.Float)
    probs_json = db.Column(db.Text)  # JSON string
    heatmap_path = db.Column(db.String(512), nullable=True)
    timestamp = db.Column(db.DateTime, default=datetime.datetime.utcnow)
    location = db.Column(db.String(128), nullable=True)  # optional location used for weather

    def to_dict(self):
        return {
            "id": self.id,
            "filename": self.filename,
            "predicted_class": self.predicted_class,
            "confidence": self.confidence,
            "probs": eval(self.probs_json),
            "heatmap": self.heatmap_path,
            "timestamp": self.timestamp.isoformat(),
            "location": self.location
        }


# ========== UTIL ==========
def allowed_file(filename):
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS


def load_trained_model():
    global model, IMG_SIZE
    print("Loading model from:", MODEL_PATH)
    # If the SavedModel dir doesn't exist, try to convert a nearby .h5 automatically
    if not os.path.exists(MODEL_PATH):
        print("SavedModel path not found:", MODEL_PATH)
        h5_candidates = [os.path.join("models", "leaf_mobilenetv2_best.h5"), "leaf_mobilenetv2_best.h5"]
        found_h5 = None
        for p in h5_candidates:
            if os.path.exists(p):
                found_h5 = p
                break
        if found_h5:
            print("Found HDF5 model:", found_h5, "— attempting conversion to SavedModel.")
            try:
                import convert_h5_to_saved as converter
                ok = converter.main()
                if ok:
                    print("Conversion succeeded; will attempt to load SavedModel now.")
                else:
                    print("Conversion attempted but reported failure. Proceeding to load (may error).")
            except Exception as conv_e:
                print("Converter import/execute failed:", conv_e)

    try:
        # first attempt (no compile) — safer for mismatched h5 files
        model = tf.keras.models.load_model(MODEL_PATH, compile=False)
    except Exception as e:
        print("First load attempt failed, trying fallback load (compile=True). Error:", e)
        # fallback: try normal load (may re-raise)
        model = tf.keras.models.load_model(MODEL_PATH)

    input_shape = model.input_shape
    if len(input_shape) != 4:
        raise ValueError("Unexpected model.input_shape: %s" % str(input_shape))
    IMG_SIZE = (input_shape[1], input_shape[2])
    print("Using IMG_SIZE:", IMG_SIZE)


def prepare_image(img_path):
    img = load_img(img_path, target_size=IMG_SIZE)
    arr = img_to_array(img) / 255.0
    arr = np.expand_dims(arr, axis=0)
    return arr


# ========== Grad-CAM ==========
def make_gradcam_heatmap(img_array, model, last_conv_layer_name=None):
    """
    Returns heatmap (H,W) in range [0,1] for input img_array.
    Assumes model outputs softmax and has convolutional backbone.
    If last_conv_layer_name not provided, tries to find the last 2D conv layer.
    """
    # find last conv layer if not provided
    if last_conv_layer_name is None:
        for layer in reversed(model.layers):
            if hasattr(layer, "output_shape"):
                shape = getattr(layer, "output_shape")
                if isinstance(shape, tuple) and len(shape) == 4:
                    last_conv_layer_name = layer.name
                    break

    last_conv_layer = model.get_layer(last_conv_layer_name)
    # create grad model
    grad_model = tf.keras.models.Model(
        [model.inputs], [last_conv_layer.output, model.output]
    )
    with tf.GradientTape() as tape:
        conv_outputs, predictions = grad_model(img_array)
        top_pred_index = tf.argmax(predictions[0])
        top_class_channel = predictions[:, top_pred_index]

    grads = tape.gradient(top_class_channel, conv_outputs)
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
    conv_outputs = conv_outputs[0]
    heatmap = conv_outputs @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)
    heatmap = tf.maximum(heatmap, 0) / (tf.reduce_max(heatmap) + 1e-8)
    heatmap = heatmap.numpy()
    # upsample to image size
    heatmap = np.uint8(255 * heatmap)
    heatmap_img = Image.fromarray(heatmap).resize(IMG_SIZE, Image.BILINEAR)
    heatmap_arr = np.array(heatmap_img) / 255.0
    return heatmap_arr


def save_gradcam_overlay(source_path, heatmap_arr, out_path, alpha=0.4):
    src = Image.open(source_path).convert("RGBA").resize(IMG_SIZE)
    heat_rgb = Image.fromarray(np.uint8(255 * np.stack([heatmap_arr, np.zeros_like(heatmap_arr), 1 - heatmap_arr], axis=2)))
    heat_rgb = heat_rgb.convert("RGBA").resize(IMG_SIZE)
    overlay = Image.blend(src, heat_rgb, alpha=alpha)
    overlay.save(out_path)
    return out_path


# ========== Weather + Risk ==========
def fetch_weather_risk(lat=None, lon=None, city=None):
    """
    Fetch simple weather and compute a naive disease risk score.
    Requires OPENWEATHER_API_KEY environment variable.
    """
    if not OPENWEATHER_API_KEY:
        return {"error": "No API key set for OpenWeatherMap."}

    base = "https://api.openweathermap.org/data/2.5/weather"
    params = {"appid": OPENWEATHER_API_KEY, "units": "metric"}
    if city:
        params["q"] = city
    elif lat and lon:
        params["lat"] = lat
        params["lon"] = lon
    else:
        return {"error": "No location provided."}

    r = requests.get(base, params=params, timeout=8)
    if r.status_code != 200:
        return {"error": f"Weather API error: {r.text}"}
    data = r.json()
    # compute a naive risk: higher risk when humidity > 75 or temp in favorable range 20-30
    hum = data.get("main", {}).get("humidity", 50)
    temp = data.get("main", {}).get("temp", 25)
    risk = 0.0
    if hum >= 80:
        risk += 0.5
    elif hum >= 60:
        risk += 0.3
    if 18 <= temp <= 30:
        risk += 0.4
    # clamp 0..1
    risk = min(1.0, risk)
    return {"weather": {"temp": temp, "humidity": hum, "desc": data["weather"][0]["description"]}, "risk": risk}


# ========== PDF REPORT ==========
def create_pdf_report(pred: Prediction):
    """
    Creates an in-memory PDF for a prediction entry and returns bytes.
    """
    buffer = io.BytesIO()
    c = canvas.Canvas(buffer, pagesize=letter)
    width, height = letter

    # Title
    c.setFont("Helvetica-Bold", 18)
    c.drawCentredString(width/2, height - 50, "Plant Leaf Disease Report")

    # Metadata
    c.setFont("Helvetica", 10)
    c.drawString(40, height - 80, f"Generated: {datetime.datetime.utcnow().isoformat()} UTC")
    c.drawString(40, height - 95, f"Prediction ID: {pred.id}")
    c.drawString(40, height - 110, f"File: {pred.filename}")
    c.drawString(40, height - 125, f"Predicted: {pred.predicted_class} ({pred.confidence*100:.2f}%)")

    # Image
    img_path = os.path.join(app.config["UPLOAD_FOLDER"], pred.filename)
    if os.path.exists(img_path):
        # place image
        c.drawImage(img_path, 40, height - 380, width=240, height=240, preserveAspectRatio=True)

    # Heatmap if exists
    if pred.heatmap_path:
        hm_path = pred.heatmap_path
        if os.path.exists(hm_path):
            c.drawImage(hm_path, 320, height - 380, width=240, height=240, preserveAspectRatio=True)

    # Probabilities
    c.setFont("Helvetica-Bold", 12)
    c.drawString(40, height - 410, "Class probabilities:")
    c.setFont("Helvetica", 10)
    probs = eval(pred.probs_json)
    y = height - 430
    for cls, p in probs.items():
        c.drawString(60, y, f"{cls}: {p*100:.2f}%")
        y -= 14

    # Footer / tips
    c.setFont("Helvetica-Bold", 12)
    c.drawString(40, y - 10, "Notes:")
    c.setFont("Helvetica", 10)
    c.drawString(60, y - 28, "This report is for demonstration only. For field decisions, consult an agronomist.")

    c.showPage()
    c.save()
    buffer.seek(0)
    return buffer


# ========== PREDICTION ROUTES ==========
@app.route("/", methods=["GET", "POST"])
def index():
    # Always fetch last 5 prediction history entries
    history_items = Prediction.query.order_by(Prediction.timestamp.desc()).limit(5).all()

    # Default context — prevents template errors on GET
    context = {
        "error": None,
        "history": history_items,
        "file_path": None,
        "predicted_class": None,
        "confidence": 0.0,            # <-- numeric default, not None
        "probs": None,
        "low_confidence": False,
        "disease_details": None,
        "heatmap_url": None,
        "prediction_id": None,
        "weather_info": None,
        "risk": None
    }

    if request.method == "POST":
        file = request.files.get("file")

        if not file:
            context["error"] = "Please upload an image."
            return render_template("index.html", **context)

        # Ensure safe file name
        safe_name = f"{int(datetime.datetime.utcnow().timestamp())}_{file.filename}"
        upload_path = os.path.join(app.config.get("UPLOAD_FOLDER", "static/uploads"), safe_name)
        os.makedirs(os.path.dirname(upload_path), exist_ok=True)
        file.save(upload_path)

        # Prepare & predict
        img_array = prepare_image(upload_path)
        preds = model.predict(img_array)[0]
        predicted_idx = int(np.argmax(preds))
        predicted_class = CLASS_NAMES[predicted_idx]
        confidence = float(preds[predicted_idx])

        # Probability dictionary
        probs = {CLASS_NAMES[i]: float(preds[i]) for i in range(len(CLASS_NAMES))}

        # Generate Grad-CAM heatmap (if available)
        heatmap_path = None
        try:
            heatmap_arr = make_gradcam_heatmap(img_array, model)
            heatmap_name = f"hm_{safe_name}"
            heatmap_dir = os.path.join(app.config.get("UPLOAD_FOLDER", "static/uploads"), "heatmaps")
            os.makedirs(heatmap_dir, exist_ok=True)
            heatmap_path = os.path.join(heatmap_dir, heatmap_name)
            save_gradcam_overlay(upload_path, heatmap_arr, heatmap_path)
        except Exception as e:
            print("Grad-CAM generation failed:", e)
            heatmap_path = None

        # Weather (optional) - expects form field or none
        location = request.form.get("location") or None
        weather_info = None
        risk = None
        if location:
            try:
                w = fetch_weather_risk(city=location)
                if "risk" in w:
                    weather_info = w.get("weather")
                    risk = w.get("risk")
            except Exception as e:
                print("Weather fetch error:", e)

        # Save to DB
        entry = Prediction(
            filename=safe_name,
            predicted_class=predicted_class,
            confidence=confidence,
            probs_json=str(probs),
            heatmap_path=heatmap_path,
            timestamp=datetime.datetime.utcnow(),
            location=location
        )
        db.session.add(entry)
        db.session.commit()

        # Fill context for template
        context.update({
            "file_path": safe_name,
            "predicted_class": predicted_class,
            "confidence": confidence,
            "probs": probs,
            "heatmap_url": heatmap_path,
            "prediction_id": entry.id,
            "weather_info": weather_info,
            "risk": risk,
            "low_confidence": (confidence < 0.40)
        })

    return render_template("index.html", **context)




@app.route("/download_report/<int:pred_id>")
def download_report(pred_id):
    pred = Prediction.query.get_or_404(pred_id)
    pdf_buf = create_pdf_report(pred)
    # send as attachment
    return send_file(pdf_buf, as_attachment=True, download_name=f"report_{pred.id}.pdf", mimetype="application/pdf")


@app.route("/api/predict", methods=["POST"])
def api_predict():
    if "file" not in request.files:
        return jsonify({"error": "No file uploaded."}), 400
    file = request.files["file"]
    if file.filename == "" or not allowed_file(file.filename):
        return jsonify({"error": "Invalid file."}), 400
    safe_name = f"{int(datetime.datetime.utcnow().timestamp())}_{file.filename}"
    save_path = os.path.join(app.config["UPLOAD_FOLDER"], safe_name)
    file.save(save_path)
    img_arr = prepare_image(save_path)
    preds = model.predict(img_arr)[0]
    idx = int(np.argmax(preds))
    conf = float(np.max(preds))
    pred_class = CLASS_NAMES[idx]

    # Try to generate Grad-CAM heatmap and save overlay
    heatmap_url = None
    heatmap_relpath = None
    try:
        heatmap_arr = make_gradcam_heatmap(img_arr, model)
        heatmap_name = f"hm_{safe_name}"
        heatmap_dir = os.path.join(app.config.get("UPLOAD_FOLDER", "static/uploads"), "heatmaps")
        os.makedirs(heatmap_dir, exist_ok=True)
        heatmap_path = os.path.join(heatmap_dir, heatmap_name)
        save_gradcam_overlay(save_path, heatmap_arr, heatmap_path)
        # URL for frontend
        heatmap_relpath = os.path.join('uploads', 'heatmaps', heatmap_name).replace('\\','/')
        heatmap_url = url_for('static', filename=heatmap_relpath)
    except Exception as e:
        print("Grad-CAM generation failed (api):", e)

    # save prediction in DB (store relative heatmap path if available)
    entry = Prediction(
        filename=safe_name,
        predicted_class=pred_class,
        confidence=conf,
        probs_json=str({CLASS_NAMES[i]: float(preds[i]) for i in range(len(CLASS_NAMES))}),
        heatmap_path=heatmap_relpath
    )
    db.session.add(entry)
    db.session.commit()

    report_url = url_for('download_report', pred_id=entry.id)

    return jsonify({
        "predicted_class": pred_class,
        "confidence": conf,
        "probabilities": {CLASS_NAMES[i]: float(preds[i]) for i in range(len(CLASS_NAMES))},
        "prediction_id": entry.id,
        "heatmap_url": heatmap_url,
        "report_url": report_url
    })


# Dashboard data endpoint
@app.route("/api/stats")
def api_stats():
    total = Prediction.query.count()
    # top classes
    rows = db.session.query(Prediction.predicted_class, db.func.count(Prediction.predicted_class))\
        .group_by(Prediction.predicted_class).all()
    class_counts = {r[0]: int(r[1]) for r in rows}
    recent = [p.to_dict() for p in Prediction.query.order_by(Prediction.timestamp.desc()).limit(20).all()]
    return jsonify({"total": total, "class_counts": class_counts, "recent": recent})


@app.route("/dashboard")
def dashboard():
    # renders template with Chart.js that calls /api/stats
    return render_template("dashboard.html")


@app.route("/simple_ui")
def simple_ui():
    return render_template("simple_ui.html")


if __name__ == "__main__":
    # initialize DB inside app context
    with app.app_context():
        db.create_all()

    load_trained_model()
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 5000)), debug=True)
