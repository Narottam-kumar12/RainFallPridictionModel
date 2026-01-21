import pandas as pd
import os
import logging
from sklearn.preprocessing import LabelEncoder


# Ensure logs directory exists
log_dir = "logs"
os.makedirs(log_dir, exist_ok=True)

# Logging configuration
logger = logging.getLogger("data_preprocessing")
logger.setLevel(logging.DEBUG)

console_handler = logging.StreamHandler()
file_handler = logging.FileHandler(os.path.join(log_dir, "data_preprocessing.log"))

formatter = logging.Formatter(
    "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)

console_handler.setFormatter(formatter)
file_handler.setFormatter(formatter)

logger.addHandler(console_handler)
logger.addHandler(file_handler)


def load_data(data_path: str) -> pd.DataFrame:
    """Load training data."""
    try:
        df = pd.read_csv(data_path)
        logger.debug("Data loaded from %s", data_path)
        return df
    except Exception as e:
        logger.error("Error loading data: %s", e)
        raise


def handle_missing_values(df: pd.DataFrame) -> pd.DataFrame:
    """Handle missing values (Pandas 3.0 safe)."""
    try:
        numeric_cols = df.select_dtypes(include=["int64", "float64"]).columns
        categorical_cols = df.select_dtypes(include=["object"]).columns

        # Fill numeric columns with median
        df[numeric_cols] = df[numeric_cols].fillna(
            df[numeric_cols].median()
        )

        # Fill categorical columns with mode
        for col in categorical_cols:
            df[col] = df[col].fillna(df[col].mode()[0])

        logger.debug("Missing values handled successfully")
        return df

    except Exception as e:
        logger.error("Error handling missing values: %s", e)
        raise


def encode_categorical_features(df: pd.DataFrame) -> pd.DataFrame:
    """Encode categorical variables."""
    try:
        categorical_cols = df.select_dtypes(include=["object"]).columns
        label_encoder = LabelEncoder()

        for col in categorical_cols:
            df[col] = label_encoder.fit_transform(df[col])

        logger.debug("Categorical features encoded")
        return df

    except Exception as e:
        logger.error("Error encoding categorical features: %s", e)
        raise


def save_processed_data(df: pd.DataFrame, data_path: str) -> None:
    """Save processed dataset."""
    try:
        processed_path = os.path.join(data_path, "processed")
        os.makedirs(processed_path, exist_ok=True)

        df.to_csv(
            os.path.join(processed_path, "train_processed.csv"),
            index=False
        )

        logger.debug("Processed data saved to %s", processed_path)

    except Exception as e:
        logger.error("Error saving processed data: %s", e)
        raise


def main():
    try:
        train_data_path = "./data/raw/train.csv"

        df = load_data(train_data_path)
        df = handle_missing_values(df)
        df = encode_categorical_features(df)

        save_processed_data(df, data_path="./data")

        logger.debug("Data preprocessing completed successfully")

    except Exception as e:
        logger.error("Data preprocessing failed: %s", e)
        print(f"Error: {e}")


if __name__ == "__main__":
    main()