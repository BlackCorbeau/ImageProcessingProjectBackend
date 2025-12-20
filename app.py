import os
import cv2
from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
from werkzeug.utils import secure_filename

from utils.face_mask import FaceMaskDetectionPipeline

# -------------------- CONFIG --------------------

UPLOAD_FOLDER = "images"
MODELS_FOLDER = "models"
DATASET_FOLDER = "./Face Mask Dataset"

ALLOWED_EXTENSIONS = {"png", "jpg", "jpeg"}

os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(MODELS_FOLDER, exist_ok=True)

# -------------------- APP --------------------

app = Flask(__name__)
CORS(app)

app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER
app.config["MAX_CONTENT_LENGTH"] = 10 * 1024 * 1024  # 10 MB

# -------------------- INIT MODELS --------------------

print("🔄 Инициализация моделей...")

detection = FaceMaskDetectionPipeline(DATASET_FOLDER)

model_files = [
    f"{MODELS_FOLDER}/hog_svm.pkl",
    f"{MODELS_FOLDER}/lbp_rf.pkl",
    f"{MODELS_FOLDER}/cnn_model.h5",
]

if all(os.path.exists(m) for m in model_files):
    detection.load_models(MODELS_FOLDER)
    print("✅ Модели загружены")
else:
    raise RuntimeError(
        "❌ Модели не найдены. Сначала обучи и сохрани модели в папку models/"
    )

# -------------------- HELPERS --------------------

def allowed_file(filename: str) -> bool:
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS

# -------------------- ROUTES --------------------

@app.route("/")
def index():
    """
    Отдаёт фронт, если index.html лежит рядом с app.py
    """
    return send_from_directory(".", "index.html")


@app.route("/api/image", methods=["POST"])
def analyze_image():
    if "file" not in request.files:
        return jsonify({"error": "File not provided"}), 400

    file = request.files["file"]

    if file.filename == "":
        return jsonify({"error": "Empty filename"}), 400

    if not allowed_file(file.filename):
        return jsonify({"error": "Unsupported file type"}), 400

    filename = secure_filename(file.filename)
    filepath = os.path.join(app.config["UPLOAD_FOLDER"], filename)

    file.save(filepath)

    img = cv2.imread(filepath)
    if img is None:
        os.remove(filepath)
        return jsonify({"error": "Invalid image"}), 400

    try:
        result = detection.analyze_image(filepath)
    except Exception as e:
        return jsonify({"error": str(e)}), 500
    finally:
        # можно удалить файл после анализа
        os.remove(filepath)

    return jsonify(result)


@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "ok"})


# -------------------- MAIN --------------------

if __name__ == "__main__":
    print("🚀 Flask API запущен")
    app.run(
        host="0.0.0.0",
        port=5000,
        debug=False,        # ВАЖНО: без двойного запуска
        use_reloader=False
    )
