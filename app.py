from flask import Flask, request, jsonify
from sys import exit
import os
from utils.face_mask import FaceMaskDetectionPipeline

UPLOAD_FOLDER = 'images'
MODELS_FOLDER = 'models'

for folder in [UPLOAD_FOLDER, MODELS_FOLDER]:
    if not os.path.exists(folder):
        os.makedirs(folder)


def initialize_models():
    detection = FaceMaskDetectionPipeline("./Face Mask Dataset")

    detection.load_dataset()

    model_files = [
        f"{MODELS_FOLDER}/hog_svm.pkl",
        f"{MODELS_FOLDER}/lbp_rf.pkl",
        f"{MODELS_FOLDER}/cnn_model.h5"
    ]

    all_models_exist = all(os.path.exists(model_file) for model_file in model_files)

    if all_models_exist:
        print("Загрузка существующих моделей...")
        try:
            detection.load_models(MODELS_FOLDER)
            print("Модели успешно загружены!")
        except Exception as e:
            print(f"Ошибка при загрузке моделей: {e}")
            print("Обучаем модели заново...")
            train_models(detection)
    else:
        print("Обучение моделей...")
        train_models(detection)

    return detection


def train_models(detection):
    print("Обучение HOG + SVM...")
    detection.train_hog_svm()

    print("Обучение LBP + RandomForest...")
    detection.train_lbp_rf()

    print("Обучение CNN...")
    detection.train_cnn(epochs=5)

    print("Сохранение моделей...")
    detection.save_models(MODELS_FOLDER)
    print("Модели успешно обучены и сохранены!")

detection = initialize_models()

app = Flask(__name__)

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

@app.route('/api/image', methods = ['GET'])
def alalyzeImage():
    if 'file' not in request.files:
        return "Not a File", 400
    file = request.files['file']
    if file.filename == '':
        return 'Название файла не валидно', 400
    if file:
        filename = file.filename
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)
        return jsonify(detection.analyze_image(filepath))
