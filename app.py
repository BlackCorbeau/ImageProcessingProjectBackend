from flask import Flask, request, jsonify
import os
from utils.face_mask import FaceMaskDetectionPipeline

UPLOAD_FOLDER = 'images'
MODELS_FOLDER = 'models'

for folder in [UPLOAD_FOLDER, MODELS_FOLDER]:
    os.makedirs(folder, exist_ok=True)

detection = None

def initialize_models():
    """Инициализация моделей - выполняется один раз"""
    global detection
    
    if detection is not None:
        return detection
        
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
    """Обучение моделей"""
    print("Обучение HOG + SVM...")
    detection.train_hog_svm()

    print("Обучение LBP + RandomForest...")
    detection.train_lbp_rf()

    print("Обучение CNN...")
    detection.train_cnn(epochs=5)

    print("Сохранение моделей...")
    detection.save_models(MODELS_FOLDER)
    print("Модели успешно обучены и сохранены!")


app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Инициализируем модели один раз при запуске приложения
@app.before_first_request
def before_first_request():
    """Выполняется один раз перед первым запросом"""
    initialize_models()


@app.route('/api/image', methods=['POST'])
def analyzeImage():
    """Обработка загруженного изображения"""
    if 'file' not in request.files:
        return "Not a File", 400
    
    file = request.files['file']
    
    if file.filename == '':
        return 'Название файла не валидно', 400
    
    if file:
        filename = file.filename
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)
        
        # Проверяем, инициализирован ли детектор
        global detection
        if detection is None:
            detection = initialize_models()
            
        return jsonify(detection.analyze_image(filepath))


if __name__ == '__main__':
    # Только для локального запуска (не через gunicorn)
    initialize_models()
    app.run(debug=True, host='0.0.0.0')
