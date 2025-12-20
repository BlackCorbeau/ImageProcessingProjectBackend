from flask import Flask, request, jsonify
import os
import threading
from utils.face_mask import FaceMaskDetectionPipeline

UPLOAD_FOLDER = 'images'
MODELS_FOLDER = 'models'

# Создаем папки один раз при запуске
for folder in [UPLOAD_FOLDER, MODELS_FOLDER]:
    os.makedirs(folder, exist_ok=True)

# Глобальные переменные (будут разделяться при использовании --preload)
detection = None
init_lock = threading.Lock()

def initialize_models_once():
    """Инициализация моделей один раз для всех воркеров"""
    global detection
    
    with init_lock:
        if detection is not None:
            return detection
            
        print("Инициализация моделей детекции масок (один раз для всех воркеров)...")
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
            print("Обучение моделей (может занять время)...")
            train_models(detection)
        
        return detection


def train_models(detection):
    """Обучение моделей с блокировкой файлов"""
    import fcntl
    
    # Блокировка файла для предотвращения одновременного обучения
    lock_file = os.path.join(MODELS_FOLDER, 'training.lock')
    fd = os.open(lock_file, os.O_CREAT | os.O_RDWR)
    
    try:
        # Пытаемся получить эксклюзивную блокировку
        fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        
        print("Обучение HOG + SVM...")
        detection.train_hog_svm()

        print("Обучение LBP + RandomForest...")
        detection.train_lbp_rf()

        print("Обучение CNN...")
        detection.train_cnn(epochs=5)

        print("Сохранение моделей...")
        detection.save_models(MODELS_FOLDER)
        print("Модели успешно обучены и сохранены!")
        
    except BlockingIOError:
        # Другой процесс уже обучает модели, ждем
        print("Другой процесс обучает модели, ожидание...")
        fcntl.flock(fd, fcntl.LOCK_EX)  # Ждем блокировку
        # После получения блокировки проверяем, не созданы ли уже модели
        model_files = [
            f"{MODELS_FOLDER}/hog_svm.pkl",
            f"{MODELS_FOLDER}/lbp_rf.pkl",
            f"{MODELS_FOLDER}/cnn_model.h5"
        ]
        if all(os.path.exists(model_file) for model_file in model_files):
            print("Модели уже обучены другим процессом, загружаем...")
            detection.load_models(MODELS_FOLDER)
        else:
            print("Модели еще не обучены, обучаем...")
            train_models(detection)  # Рекурсивный вызов после получения блокировки
    finally:
        fcntl.flock(fd, fcntl.LOCK_UN)
        os.close(fd)


# Инициализируем модели один раз при импорте модуля
# (это сработает только с --preload в gunicorn)
detection = initialize_models_once()

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

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
        
        # Используем уже инициализированный детектор
        return jsonify(detection.analyze_image(filepath))


@app.route('/health', methods=['GET'])
def health_check():
    """Проверка работоспособности сервиса"""
    return jsonify({"status": "ok", "message": "Service is running"})


if __name__ == '__main__':
    # Только для локального запуска
    print("Запуск Flask приложения...")
    app.run(debug=True, host='0.0.0.0', port=5000)
