import pandas as pd
import os
import logging
import joblib

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score
)

from xgboost import XGBClassifier


# Ensure logs directory exists
log_dir = "logs"
os.makedirs(log_dir, exist_ok=True)

# Logging configuration
logger = logging.getLogger("model_building")
logger.setLevel(logging.DEBUG)

console_handler = logging.StreamHandler()
file_handler = logging.FileHandler(os.path.join(log_dir, "model_building.log"))

formatter = logging.Formatter(
    "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)

console_handler.setFormatter(formatter)
file_handler.setFormatter(formatter)

logger.addHandler(console_handler)
logger.addHandler(file_handler)


def load_feature_data(data_path: str) -> pd.DataFrame:
    """Load feature engineered dataset."""
    try:
        df = pd.read_csv(data_path)
        logger.debug("Feature data loaded from %s", data_path)
        return df
    except Exception as e:
        logger.error("Error loading feature data: %s", e)
        raise


def split_features_target(df: pd.DataFrame):
    """Split dataset into X and y."""
    try:
        X = df.drop(columns=["target"])
        y = df["target"]

        logger.debug("Features and target separated")
        return X, y
    except KeyError as e:
        logger.error("Target column missing: %s", e)
        raise


def evaluate_model(model, X_test, y_test, model_name: str):
    """Evaluate trained model."""
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


def train_random_forest(X_train, y_train):
    """Train Random Forest model."""
    model = RandomForestClassifier(
        n_estimators=200,
        max_depth=10,
        random_state=42,
        class_weight="balanced"
    )
    model.fit(X_train, y_train)
    logger.debug("Random Forest training completed")
    return model


def train_xgboost(X_train, y_train):
    """Train XGBoost model."""
    model = XGBClassifier(
        n_estimators=300,
        max_depth=6,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        eval_metric="logloss",
        random_state=42
    )
    model.fit(X_train, y_train)
    logger.debug("XGBoost training completed")
    return model


def save_model(model, model_name: str):
    """Save trained model."""
    try:
        model_dir = "./models"
        os.makedirs(model_dir, exist_ok=True)

        model_path = os.path.join(model_dir, model_name)
        joblib.dump(model, model_path)

        logger.debug("Model saved at %s", model_path)
    except Exception as e:
        logger.error("Error saving model: %s", e)
        raise


def main():
    try:
        feature_data_path = "/home/demo3/RainFallPridictionModel/data/features/train_features.csv"

        df = load_feature_data(feature_data_path)
        X, y = split_features_target(df)

        X_train, X_test, y_train, y_test = train_test_split(
            X,
            y,
            test_size=0.2,
            random_state=42,
            stratify=y
        )

        # ---------------- Random Forest ----------------
        rf_model = train_random_forest(X_train, y_train)
        rf_f1 = evaluate_model(rf_model, X_test, y_test, "Random Forest")

        # ---------------- XGBoost ----------------
        xgb_model = train_xgboost(X_train, y_train)
        xgb_f1 = evaluate_model(xgb_model, X_test, y_test, "XGBoost")

        # ---------------- Model Selection ----------------
        if xgb_f1 >= rf_f1:
            save_model(xgb_model, "rainfall_xgboost_model.pkl")
            logger.debug("XGBoost selected as final model")
        else:
            save_model(rf_model, "rainfall_random_forest_model.pkl")
            logger.debug("Random Forest selected as final model")

        logger.debug("Model building pipeline completed successfully")

    except Exception as e:
        logger.error("Model building failed: %s", e)
        print(f"Error: {e}")


if __name__ == "__main__":
    main()