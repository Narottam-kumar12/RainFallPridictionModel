import pandas as pd
import os
import logging
from sklearn.preprocessing import LabelEncoder
import pickle  # To save encoder for future use

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
    """Load dataset from CSV."""
    try:
        df = pd.read_csv(data_path)
        logger.debug("Data loaded from %s", data_path)
        return df
    except Exception as e:
        logger.error("Error loading data: %s", e)
        raise


def handle_missing_values(df: pd.DataFrame) -> pd.DataFrame:
    """Handle missing values."""
    try:
        numeric_cols = df.select_dtypes(include=["int64", "float64"]).columns
        categorical_cols = df.select_dtypes(include=["object"]).columns

        # Fill numeric columns with median
        df[numeric_cols] = df[numeric_cols].fillna(df[numeric_cols].median())

        # Fill categorical columns with mode
        for col in categorical_cols:
            df[col] = df[col].fillna(df[col].mode()[0])

        logger.debug("Missing values handled successfully")
        return df

    except Exception as e:
        logger.error("Error handling missing values: %s", e)
        raise


def encode_categorical_features_train(df: pd.DataFrame) -> (pd.DataFrame, dict):
    """Encode categorical variables for train set and save encoders."""
    try:
        categorical_cols = df.select_dtypes(include=["object"]).columns
        encoders = {}

        for col in categorical_cols:
            le = LabelEncoder()
            df[col] = le.fit_transform(df[col])
            encoders[col] = le  # Save encoder for test set

        # Save encoders
        os.makedirs("encoders", exist_ok=True)
        with open("encoders/label_encoders.pkl", "wb") as f:
            pickle.dump(encoders, f)

        logger.debug("Categorical features encoded for train set")
        return df, encoders

    except Exception as e:
        logger.error("Error encoding categorical features: %s", e)
        raise


def encode_categorical_features_test(df: pd.DataFrame, encoders: dict) -> pd.DataFrame:
    """Encode test set using saved train encoders."""
    try:
        for col, le in encoders.items():
            if col in df.columns:
                # Handle unseen categories by assigning -1
                df[col] = df[col].map(lambda x: le.transform([x])[0] if x in le.classes_ else -1)

        logger.debug("Categorical features encoded for test set")
        return df

    except Exception as e:
        logger.error("Error encoding categorical features for test: %s", e)
        raise


def save_processed_data(df: pd.DataFrame, data_path: str, file_name: str) -> None:
    """Save processed dataset."""
    try:
        processed_path = os.path.join(data_path, "processed")
        os.makedirs(processed_path, exist_ok=True)

        df.to_csv(os.path.join(processed_path, file_name), index=False)
        logger.debug("Processed data saved to %s", os.path.join(processed_path, file_name))

    except Exception as e:
        logger.error("Error saving processed data: %s", e)
        raise


def main():
    try:
        # Paths
        train_data_path = "./data/raw/train.csv"
        test_data_path = "./data/raw/test.csv"
        save_dir = "./data"

        # --- Process train data ---
        train_df = load_data(train_data_path)
        train_df = handle_missing_values(train_df)
        train_df, encoders = encode_categorical_features_train(train_df)
        save_processed_data(train_df, save_dir, "train_processed.csv")

        # --- Process test data ---
        test_df = load_data(test_data_path)
        test_df = handle_missing_values(test_df)
        test_df = encode_categorical_features_test(test_df, encoders)
        save_processed_data(test_df, save_dir, "test_processed.csv")

        logger.debug("Data preprocessing completed successfully for both train and test datasets")

    except Exception as e:
        logger.error("Data preprocessing failed: %s", e)
        print(f"Error: {e}")


if __name__ == "__main__":
    main()
