from face_mask import FaceMaskDetectionPipeline
import os


DATASET_PATH = "Face Mask Dataset"
TEST_IMAGE_PATH = "Face Mask Dataset/Test/WithMask/3.png"

MODELS_PATH = "models"


def train_and_save():
    print("=== TRAINING MODELS ===")

    pipeline = FaceMaskDetectionPipeline(DATASET_PATH)

    pipeline.load_dataset(use_validation=True)

    pipeline.train_hog_svm()
    pipeline.train_lbp_rf()
    pipeline.train_cnn(epochs=5)

    pipeline.save_models(MODELS_PATH)

    print("=== TRAINING FINISHED ===\n")



def test_inference():
    print("=== TESTING INFERENCE ===")

    pipeline = FaceMaskDetectionPipeline()
    pipeline.load_models(MODELS_PATH)

    if not os.path.exists(TEST_IMAGE_PATH):
        print("❌ Test image not found:", TEST_IMAGE_PATH)
        return

    result = pipeline.analyze_image(TEST_IMAGE_PATH)

    print("\n--- RESULTS ---")
    for model_name, data in result.items():
        if model_name != "best_model":
            print(
                f"{model_name}: "
                f"{data['prediction']} "
                f"(confidence={data['confidence']:.2f})"
            )

    print("\nBest model:", result["best_model"])
    print("=== TEST FINISHED ===")



if __name__ == "__main__":

    if not os.path.exists(MODELS_PATH):
        train_and_save()
    else:
        print("Models already exist, skipping training\n")

    test_inference()
