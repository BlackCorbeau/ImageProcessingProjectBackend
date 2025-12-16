import os
import cv2
import numpy as np
import joblib
import tensorflow as tf

from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report

from skimage.feature import hog, local_binary_pattern
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D
from tensorflow.keras.models import Model


class FaceMaskDetectionPipeline:
    def __init__(self, dataset_path=None, img_size=224):
        self.dataset_path = dataset_path
        self.img_size = img_size

        self.X = []
        self.y = []

        self.hog_svm = None
        self.lbp_rf = None
        self.cnn_model = None

        np.random.seed(42)
        tf.random.set_seed(42)

    # 1. Загрузка и исследование датасета
    def load_dataset(self, use_validation=False):
        splits = ["Train"]
        if use_validation:
            splits.append("Validation")

        classes = {
            "WithMask": 0,
            "WithoutMask": 1
        }

        for split in splits:
            split_path = os.path.join(self.dataset_path, split)

            for class_name, label in classes.items():
                class_path = os.path.join(split_path, class_name)

                for file in os.listdir(class_path):
                    img_path = os.path.join(class_path, file)
                    img = cv2.imread(img_path)

                    if img is None:
                        continue

                    img = cv2.resize(img, (self.img_size, self.img_size))
                    self.X.append(img)
                    self.y.append(label)

        self.X = np.array(self.X)
        self.y = np.array(self.y)

        print("Dataset loaded")
        print("Total images:", len(self.X))
        print("With mask:", np.sum(self.y == 0))
        print("Without mask:", np.sum(self.y == 1))

    # 2. HOG + SVM
    def train_hog_svm(self):
        features = []

        for img in self.X:
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            feat = hog(
                gray,
                pixels_per_cell=(16, 16),
                cells_per_block=(2, 2),
                feature_vector=True
            )
            features.append(feat)

        X_feat = np.array(features)
        X_train, X_test, y_train, y_test = train_test_split(
            X_feat, self.y, test_size=0.2, random_state=42
        )

        self.hog_svm = SVC(kernel="linear", probability=True)
        self.hog_svm.fit(X_train, y_train)

        preds = self.hog_svm.predict(X_test)
        print("HOG + SVM results:")
        print(classification_report(y_test, preds))

    # 3. LBP + RandomForest
    def train_lbp_rf(self):
        features = []

        for img in self.X:
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            lbp = local_binary_pattern(gray, P=8, R=1, method="uniform")
            hist, _ = np.histogram(lbp, bins=256, range=(0, 256))
            features.append(hist)

        X_feat = np.array(features)
        X_train, X_test, y_train, y_test = train_test_split(
            X_feat, self.y, test_size=0.2, random_state=42
        )

        self.lbp_rf = RandomForestClassifier(n_estimators=100)
        self.lbp_rf.fit(X_train, y_train)

        preds = self.lbp_rf.predict(X_test)
        print("LBP + RandomForest results:")
        print(classification_report(y_test, preds))

    # 4. CNN
    def train_cnn(self, epochs=5):
        X_norm = self.X / 255.0

        X_train, X_test, y_train, y_test = train_test_split(
            X_norm, self.y, test_size=0.2, random_state=42
        )

        base_model = MobileNetV2(
            weights="imagenet",
            include_top=False,
            input_shape=(self.img_size, self.img_size, 3)
        )

        for layer in base_model.layers:
            layer.trainable = False

        x = base_model.output
        x = GlobalAveragePooling2D()(x)
        output = Dense(1, activation="sigmoid")(x)

        self.cnn_model = Model(inputs=base_model.input, outputs=output)

        self.cnn_model.compile(
            optimizer="adam",
            loss="binary_crossentropy",
            metrics=["accuracy"]
        )

        self.cnn_model.fit(
            X_train,
            y_train,
            validation_data=(X_test, y_test),
            epochs=epochs,
            batch_size=32
        )

    # 5. Сохранение и загрузка моделей
    def save_models(self, path="models"):
        os.makedirs(path, exist_ok=True)

        joblib.dump(self.hog_svm, f"{path}/hog_svm.pkl")
        joblib.dump(self.lbp_rf, f"{path}/lbp_rf.pkl")
        self.cnn_model.save(f"{path}/cnn_model.h5")

    def load_models(self, path="models"):
        self.hog_svm = joblib.load(f"{path}/hog_svm.pkl")
        self.lbp_rf = joblib.load(f"{path}/lbp_rf.pkl")
        self.cnn_model = tf.keras.models.load_model(f"{path}/cnn_model.h5")

    # Анализ изображения
    def analyze_image(self, image_path):
        results = {}

        img = cv2.imread(image_path)
        img = cv2.resize(img, (self.img_size, self.img_size))
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # --- HOG + SVM ---
        hog_feat = hog(
            gray,
            pixels_per_cell=(16, 16),
            cells_per_block=(2, 2),
            feature_vector=True
        )

        prob = self.hog_svm.predict_proba([hog_feat])[0]
        pred = np.argmax(prob)

        results["hog_svm"] = {
            "prediction": "with_mask" if pred == 0 else "without_mask",
            "confidence": float(prob[pred])
        }

        # --- LBP + RF ---
        lbp = local_binary_pattern(gray, P=8, R=1, method="uniform")
        hist, _ = np.histogram(lbp, bins=256, range=(0, 256))

        prob = self.lbp_rf.predict_proba([hist])[0]
        pred = np.argmax(prob)

        results["lbp_rf"] = {
            "prediction": "with_mask" if pred == 0 else "without_mask",
            "confidence": float(prob[pred])
        }

        # --- CNN ---
        img_cnn = img / 255.0
        img_cnn = np.expand_dims(img_cnn, axis=0)

        prob = self.cnn_model.predict(img_cnn)[0][0]
        pred = 0 if prob < 0.5 else 1

        results["cnn"] = {
            "prediction": "with_mask" if pred == 0 else "without_mask",
            "confidence": float(prob if pred == 1 else 1 - prob)
        }

        best_model = max(results, key=lambda x: results[x]["confidence"])
        results["best_model"] = best_model

        return results
