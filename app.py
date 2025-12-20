from flask import Flask, request, jsonify
import os
from utils.face_mask import FaceMaskDetectionPipeline

UPLOAD_FOLDER = 'images'
MODELS_FOLDER = 'models'

# Создаем папки один раз при запуске
for folder in [UPLOAD_FOLDER, MODELS_FOLDER]:
    os.makedirs(folder, exist_ok=True)

# Глобальная переменная для хранения инициализированного детектора
detection = None
is_initialized = False

def initialize_models():
    """Инициализация моделей - выполняется один раз"""
    global detection, is_initialized
    
    if is_initialized and detection is not None:
        return detection
        
    print("Инициализация моделей детекции масок...")
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
    
    is_initialized = True
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

# Инициализируем модели при запуске приложения
with app.app_context():
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


@app.route('/health', methods=['GET'])
def health_check():
    """Проверка работоспособности сервиса"""
    return jsonify({"status": "ok", "initialized": is_initialized})


if __name__ == '__main__':
    # Только для локального запуска (не через gunicorn)
    print("Запуск Flask приложения...")
    app.run(debug=True, host='0.0.0.0', port=5000)
