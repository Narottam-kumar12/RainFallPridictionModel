import pandas as pd
import os
import logging

# Ensure logs directory exists
log_dir = "logs"
os.makedirs(log_dir, exist_ok=True)

# Logging configuration
logger = logging.getLogger("feature_engineering")
logger.setLevel(logging.DEBUG)

console_handler = logging.StreamHandler()
file_handler = logging.FileHandler(os.path.join(log_dir, "feature_engineering.log"))

formatter = logging.Formatter(
    "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)

console_handler.setFormatter(formatter)
file_handler.setFormatter(formatter)

logger.addHandler(console_handler)
logger.addHandler(file_handler)


def load_data(data_path: str) -> pd.DataFrame:
    """Load dataset from CSV."""
    try:
        df = pd.read_csv(data_path)
        logger.debug("Data loaded from %s", data_path)
        return df
    except Exception as e:
        logger.error("Error loading data: %s", e)
        raise


def create_rainfall_features(df: pd.DataFrame) -> pd.DataFrame:
    """Create meaningful features for rainfall prediction."""
    try:
        # Temperature difference
        if "MaxTemp" in df.columns and "MinTemp" in df.columns:
            df["TempRange"] = df["MaxTemp"] - df["MinTemp"]

        # Humidity average
        if "Humidity9am" in df.columns and "Humidity3pm" in df.columns:
            df["AvgHumidity"] = (df["Humidity9am"] + df["Humidity3pm"]) / 2

        # Pressure difference
        if "Pressure9am" in df.columns and "Pressure3pm" in df.columns:
            df["PressureDiff"] = df["Pressure9am"] - df["Pressure3pm"]

        # Wind speed average
        if "WindSpeed9am" in df.columns and "WindSpeed3pm" in df.columns:
            df["AvgWindSpeed"] = (df["WindSpeed9am"] + df["WindSpeed3pm"]) / 2

        logger.debug("Rainfall related features created")
        return df

    except Exception as e:
        logger.error("Error during feature creation: %s", e)
        raise


def drop_unused_features(df: pd.DataFrame, target_col: str = "Rainfall") -> pd.DataFrame:
    """Drop features that add noise (like target column)."""
    try:
        drop_cols = [target_col] if target_col in df.columns else []
        if drop_cols:
            df.drop(columns=drop_cols, inplace=True)
        logger.debug("Unused features dropped: %s", drop_cols)
        return df

    except Exception as e:
        logger.error("Error dropping unused features: %s", e)
        raise


def save_featured_data(df: pd.DataFrame, save_path: str, file_name: str) -> None:
    """Save feature engineered dataset."""
    try:
        os.makedirs(save_path, exist_ok=True)
        full_path = os.path.join(save_path, file_name)
        df.to_csv(full_path, index=False)
        logger.debug("Feature engineered data saved to %s", full_path)
    except Exception as e:
        logger.error("Error saving feature engineered data: %s", e)
        raise


def main():
    try:
        # Paths for train and test datasets
        train_data_path = "/home/demo3/RainFallPridictionModel/data/processed/train_processed.csv"
        test_data_path = "/home/demo3/RainFallPridictionModel/data/processed/test_processed.csv"
        save_dir = "./data/features"

        # Process train data
        train_df = load_data(train_data_path)
        train_df = create_rainfall_features(train_df)
        train_df = drop_unused_features(train_df)
        save_featured_data(train_df, save_dir, "train_features.csv")

        # Process test data
        test_df = load_data(test_data_path)
        test_df = create_rainfall_features(test_df)
        test_df = drop_unused_features(test_df, target_col="Rainfall")  # Drop target if exists
        save_featured_data(test_df, save_dir, "test_features.csv")

        logger.debug("Feature engineering completed successfully for both train and test datasets")

    except Exception as e:
        logger.error("Feature engineering failed: %s", e)
        print(f"Error: {e}")


if __name__ == "__main__":
    main()
