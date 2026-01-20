import os
import logging
import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from xgboost import XGBClassifier
import yaml

# =========================
# Logging Configuration
# =========================
log_dir = "logs"
os.makedirs(log_dir, exist_ok=True)

logger = logging.getLogger("model_building")
logger.setLevel(logging.DEBUG)

console_handler = logging.StreamHandler()
file_handler = logging.FileHandler(os.path.join(log_dir, "model_building.log"))

formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")

console_handler.setFormatter(formatter)
file_handler.setFormatter(formatter)

logger.addHandler(console_handler)
logger.addHandler(file_handler)


# =========================
# Load Parameters
# =========================
def load_params(stage_name: str):
    params_file = os.path.join(os.getcwd(), "params.yaml")
    with open(params_file, "r") as f:
        all_params = yaml.safe_load(f)
    return all_params.get(stage_name, {})


# =========================
# Load Feature Data
# =========================
def load_feature_data(data_path: str) -> pd.DataFrame:
    df = pd.read_csv(data_path)
    logger.debug("Feature data loaded from %s", data_path)
    return df


# =========================
# Split features and target
# =========================
def split_features_target(df: pd.DataFrame):
    X = df.drop(columns=["target"])
    y = df["target"]
    logger.debug("Features and target separated")
    return X, y


# =========================
# Model training functions
# =========================
def train_random_forest(X_train, y_train, params):
    model = RandomForestClassifier(
        n_estimators=params.get("rf_n_estimators", 200),
        max_depth=params.get("rf_max_depth", 10),
        random_state=params.get("rf_random_state", 42),
        class_weight="balanced"
    )
    model.fit(X_train, y_train)
    logger.debug("Random Forest training completed")
    return model


def train_xgboost(X_train, y_train, params):
    model = XGBClassifier(
        n_estimators=params.get("xgb_n_estimators", 300),
        max_depth=params.get("xgb_max_depth", 6),
        learning_rate=params.get("xgb_learning_rate", 0.05),
        subsample=params.get("xgb_subsample", 0.8),
        colsample_bytree=params.get("xgb_colsample_bytree", 0.8),
        eval_metric="logloss",
        random_state=params.get("xgb_random_state", 42)
    )
    model.fit(X_train, y_train)
    logger.debug("XGBoost training completed")
    return model


# =========================
# Evaluate model
# =========================
def evaluate_model(model, X_test, y_test, model_name: str):
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred)
    rec = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)

    logger.debug("Evaluation results for %s:", model_name)
    logger.debug("Accuracy  : %.4f", acc)
    logger.debug("Precision : %.4f", prec)
    logger.debug("Recall    : %.4f", rec)
    logger.debug("F1 Score  : %.4f", f1)

    return f1


# =========================
# Save model
# =========================
def save_model(model, model_name: str):
    os.makedirs("models", exist_ok=True)
    model_path = os.path.join("models", model_name)
    joblib.dump(model, model_path)
    logger.debug("Model saved at %s", model_path)
    return model_path


# =========================
# Main pipeline
# =========================
def main():
    try:
        params = load_params("model_building")

        feature_data_path = "./data/features/train_features.csv"
        df = load_feature_data(feature_data_path)
        X, y = split_features_target(df)

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )

        # Train both models
        rf_model = train_random_forest(X_train, y_train, params)
        xgb_model = train_xgboost(X_train, y_train, params)

        # Evaluate both models
        rf_f1 = evaluate_model(rf_model, X_test, y_test, "Random Forest")
        xgb_f1 = evaluate_model(xgb_model, X_test, y_test, "XGBoost")

        # Save both models
        save_model(rf_model, "rainfall_random_forest_model.pkl")
        save_model(xgb_model, "rainfall_xgboost_model.pkl")

        # Select best model for production (can be used in evaluation)
        best_model_name = "rainfall_xgboost_model.pkl" if xgb_f1 >= rf_f1 else "rainfall_random_forest_model.pkl"
        logger.debug("Selected best model: %s", best_model_name)

    except Exception as e:
        logger.error("Model building failed: %s", e)
        print(f"Error: {e}")


if __name__ == "__main__":
    main()
