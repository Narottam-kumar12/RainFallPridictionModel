import os
import logging
import pandas as pd
import joblib

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    confusion_matrix
)

# =========================
# Logging Configuration
# =========================
log_dir = "logs"
os.makedirs(log_dir, exist_ok=True)

logger = logging.getLogger("model_evaluation")
logger.setLevel(logging.DEBUG)

console_handler = logging.StreamHandler()
file_handler = logging.FileHandler(os.path.join(log_dir, "model_evaluation.log"))

formatter = logging.Formatter(
    "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)

console_handler.setFormatter(formatter)
file_handler.setFormatter(formatter)

logger.addHandler(console_handler)
logger.addHandler(file_handler)


# =========================
# Load Feature Data
# =========================
def load_test_data(feature_path: str) -> pd.DataFrame:
    try:
        df = pd.read_csv(feature_path)
        logger.debug("Test feature data loaded from %s", feature_path)
        return df
    except Exception as e:
        logger.error("Failed to load test feature data: %s", e)
        raise


# =========================
# Load Model
# =========================
def load_model(model_path: str):
    try:
        model = joblib.load(model_path)
        logger.debug("Model loaded from %s", model_path)
        return model
    except Exception as e:
        logger.error("Failed to load model: %s", e)
        raise


# =========================
# Model Evaluation
# =========================
def evaluate_model(model, X_test, y_test):
    try:
        y_pred = model.predict(X_test)

        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred)
        recall = recall_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)

        logger.debug("Model Evaluation Results:")
        logger.debug("Accuracy  : %.4f", accuracy)
        logger.debug("Precision : %.4f", precision)
        logger.debug("Recall    : %.4f", recall)
        logger.debug("F1 Score  : %.4f", f1)

        logger.debug("Classification Report:\n%s",
                     classification_report(y_test, y_pred))

        logger.debug("Confusion Matrix:\n%s",
                     confusion_matrix(y_test, y_pred))

        return {
            "accuracy": accuracy,
            "precision": precision,
            "recall": recall,
            "f1_score": f1
        }

    except Exception as e:
        logger.error("Error during model evaluation: %s", e)
        raise


# =========================
# Save Evaluation Results
# =========================
def save_metrics(metrics: dict, output_path: str):
    try:
        os.makedirs(output_path, exist_ok=True)
        metrics_df = pd.DataFrame([metrics])
        metrics_df.to_csv(
            os.path.join(output_path, "evaluation_metrics.csv"),
            index=False
        )
        logger.debug("Evaluation metrics saved at %s", output_path)
    except Exception as e:
        logger.error("Failed to save evaluation metrics: %s", e)
        raise


# =========================
# Main Pipeline
# =========================
def main():
    try:
        test_feature_path = "./data/features/test_features.csv"
        model_path = "./models/rainfall_random_forest_model.pkl"
        output_path = "./reports"

        df = load_test_data(test_feature_path)

        X_test = df.drop(columns=["target"])
        y_test = df["target"]

        logger.debug("Features and target separated for evaluation")

        model = load_model(model_path)

        metrics = evaluate_model(model, X_test, y_test)

        save_metrics(metrics, output_path)

        logger.debug("Model evaluation pipeline completed successfully")

    except Exception as e:
        logger.error("Model evaluation failed: %s", e)
        print(f"Error: {e}")


if __name__ == "__main__":
    main()