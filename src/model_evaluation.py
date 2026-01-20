import os
import logging
import pandas as pd
import joblib
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import yaml

# Logging
log_dir = "logs"
os.makedirs(log_dir, exist_ok=True)
logger = logging.getLogger("model_evaluation")
logger.setLevel(logging.DEBUG)
console_handler = logging.StreamHandler()
file_handler = logging.FileHandler(os.path.join(log_dir, "model_evaluation.log"))
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
console_handler.setFormatter(formatter)
file_handler.setFormatter(formatter)
logger.addHandler(console_handler)
logger.addHandler(file_handler)

# Params loader
def load_params(stage_name: str):
    params_file = os.path.join(os.getcwd(), "params.yaml")
    with open(params_file, "r") as f:
        all_params = yaml.safe_load(f)
    return all_params.get(stage_name, {})

def load_test_data(feature_path: str) -> pd.DataFrame:
    return pd.read_csv(feature_path)

def load_model(model_path: str):
    return joblib.load(model_path)

def evaluate_model(model, X_test, y_test):
    y_pred = model.predict(X_test)
    return {
        "accuracy": accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred),
        "recall": recall_score(y_test, y_pred),
        "f1_score": f1_score(y_test, y_pred)
    }

def save_metrics(metrics: dict, output_path: str):
    os.makedirs(output_path, exist_ok=True)
    pd.DataFrame([metrics]).to_csv(os.path.join(output_path, "evaluation_metrics.csv"), index=False)
    logger.debug("Evaluation metrics saved at %s", output_path)

def main():
    try:
        params = load_params("model_evaluation")
        test_feature_path = "./data/features/test_features.csv"
        model_path = os.path.join("models", params.get("model_choice","rainfall_random_forest_model.pkl"))
        output_path = "./reports"

        df = load_test_data(test_feature_path)
        X_test = df.drop(columns=["target"])
        y_test = df["target"]

        model = load_model(model_path)
        metrics = evaluate_model(model, X_test, y_test)
        save_metrics(metrics, output_path)

        logger.debug("Model evaluation completed successfully")
    except Exception as e:
        logger.error("Model evaluation failed: %s", e)
        print(f"Error: {e}")

if __name__ == "__main__":
    main()
