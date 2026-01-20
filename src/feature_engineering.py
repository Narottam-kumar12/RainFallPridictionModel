import pandas as pd
import os
import logging
import yaml

# Logging
log_dir = "logs"
os.makedirs(log_dir, exist_ok=True)
logger = logging.getLogger("feature_engineering")
logger.setLevel(logging.DEBUG)
console_handler = logging.StreamHandler()
file_handler = logging.FileHandler(os.path.join(log_dir, "feature_engineering.log"))
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

# Functions
def load_data(data_path: str) -> pd.DataFrame:
    return pd.read_csv(data_path)

def create_rainfall_features(df: pd.DataFrame) -> pd.DataFrame:
    if "MaxTemp" in df.columns and "MinTemp" in df.columns:
        df["TempRange"] = df["MaxTemp"] - df["MinTemp"]
    if "Humidity9am" in df.columns and "Humidity3pm" in df.columns:
        df["AvgHumidity"] = (df["Humidity9am"] + df["Humidity3pm"]) / 2
    if "Pressure9am" in df.columns and "Pressure3pm" in df.columns:
        df["PressureDiff"] = df["Pressure9am"] - df["Pressure3pm"]
    if "WindSpeed9am" in df.columns and "WindSpeed3pm" in df.columns:
        df["AvgWindSpeed"] = (df["WindSpeed9am"] + df["WindSpeed3pm"]) / 2
    logger.debug("Rainfall related features created")
    return df

def drop_unused_features(df: pd.DataFrame, target_col: str):
    if target_col in df.columns:
        df.drop(columns=[target_col], inplace=True)
    logger.debug("Unused features dropped")
    return df

def save_featured_data(df: pd.DataFrame, save_path: str, file_name: str):
    os.makedirs(save_path, exist_ok=True)
    df.to_csv(os.path.join(save_path, file_name), index=False)
    logger.debug("Feature engineered data saved to %s", os.path.join(save_path, file_name))

def main():
    try:
        params = load_params("feature_engineering")
        target_col = params.get("target_col", "Rainfall")

        train_path = "./data/processed/train_processed.csv"
        test_path = "./data/processed/test_processed.csv"
        save_dir = "./data/features"

        train_df = load_data(train_path)
        train_df = create_rainfall_features(train_df)
        train_df = drop_unused_features(train_df, target_col)
        save_featured_data(train_df, save_dir, "train_features.csv")

        test_df = load_data(test_path)
        test_df = create_rainfall_features(test_df)
        test_df = drop_unused_features(test_df, target_col)
        save_featured_data(test_df, save_dir, "test_features.csv")

        logger.debug("Feature engineering completed successfully")
    except Exception as e:
        logger.error("Feature engineering failed: %s", e)
        print(f"Error: {e}")

if __name__ == "__main__":
    main()
