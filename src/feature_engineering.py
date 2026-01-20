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


def load_processed_data(data_path: str) -> pd.DataFrame:
    """Load preprocessed dataset."""
    try:
        df = pd.read_csv(data_path)
        logger.debug("Processed data loaded from %s", data_path)
        return df
    except Exception as e:
        logger.error("Error loading processed data: %s", e)
        raise


def create_rainfall_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create meaningful features for rainfall prediction.
    """
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
            df["AvgWindSpeed"] = (
                df["WindSpeed9am"] + df["WindSpeed3pm"]
            ) / 2

        logger.debug("Rainfall related features created")
        return df

    except Exception as e:
        logger.error("Error during feature creation: %s", e)
        raise


def drop_unused_features(df: pd.DataFrame) -> pd.DataFrame:
    """Drop features that add noise."""
    try:
        drop_cols = []

        if "Rainfall" in df.columns:
            drop_cols.append("Rainfall")

        if drop_cols:
            df.drop(columns=drop_cols, inplace=True)

        logger.debug("Unused features dropped: %s", drop_cols)
        return df

    except Exception as e:
        logger.error("Error dropping unused features: %s", e)
        raise


def save_featured_data(df: pd.DataFrame, data_path: str) -> None:
    """Save feature engineered dataset."""
    try:
        feature_path = os.path.join(data_path, "features")
        os.makedirs(feature_path, exist_ok=True)

        df.to_csv(
            os.path.join(feature_path, "train_features.csv"),
            index=False
        )

        logger.debug("Feature engineered data saved to %s", feature_path)

    except Exception as e:
        logger.error("Error saving feature engineered data: %s", e)
        raise


def main():
    try:
        processed_data_path = "/home/demo3/RainFallPridictionModel/data/processed/train_processed.csv"

        df = load_processed_data(processed_data_path)
        df = create_rainfall_features(df)
        df = drop_unused_features(df)

        save_featured_data(df, data_path="./data")

        logger.debug("Feature engineering completed successfully")

    except Exception as e:
        logger.error("Feature engineering failed: %s", e)
        print(f"Error: {e}")


if __name__ == "__main__":
    main()
