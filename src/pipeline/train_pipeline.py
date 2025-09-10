import os
import sys
from src.utils import save_object
from src.components.data_transformation import DataTransformation
from src.components.model_trainer import ModelTrainer

def main():
    try:
        # ✅ CSV file paths (artifacts folder ke andar)
        train_path = os.path.join("artifacts", "train.csv")
        test_path = os.path.join("artifacts", "test.csv")

        # ✅ Data Transformation
        data_transformation = DataTransformation()
        train_arr, test_arr, preprocessor = data_transformation.initiate_data_transformation(
            train_path, test_path
        )

        # ✅ Model Training
        model_trainer = ModelTrainer()
        model = model_trainer.initiate_model_training(train_arr, test_arr)

        # ✅ Save model and preprocessor in artifacts
        model_path = os.path.join("artifacts", "model.pkl")
        preprocessor_path = os.path.join("artifacts", "preprocessor.pkl")

        save_object(file_path=model_path, obj=model)
        save_object(file_path=preprocessor_path, obj=preprocessor)

        print("✅ Training complete! Model and Preprocessor saved in artifacts folder.")

    except Exception as e:
        print(f"Error in training pipeline: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
