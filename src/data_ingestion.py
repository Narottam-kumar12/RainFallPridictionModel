import pandas as pd
import os
import logging
from sklearn.model_selection import train_test_split
import yaml

# =========================
# Logging setup
# =========================
log_dir = "logs"
os.makedirs(log_dir, exist_ok=True)
logger = logging.getLogger("data_ingestion")
logger.setLevel(logging.DEBUG)
console_handler = logging.StreamHandler()
file_handler = logging.FileHandler(os.path.join(log_dir, "data_ingestion.log"))
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
console_handler.setFormatter(formatter)
file_handler.setFormatter(formatter)
logger.addHandler(console_handler)
logger.addHandler(file_handler)

# =========================
# Load params function
# =========================
def load_params(stage_name: str):
    params_file = os.path.join(os.getcwd(), "params.yaml")
    with open(params_file, "r") as f:
        all_params = yaml.safe_load(f)
    return all_params.get(stage_name, {})

# =========================
# Functions
# =========================
def load_data(data_url: str) -> pd.DataFrame:
    df = pd.read_csv(data_url)
    logger.debug("Data loaded from %s", data_url)
    return df

def preprocess_data(df: pd.DataFrame) -> pd.DataFrame:
    df.drop(columns=['Date'], inplace=True)
    df.rename(columns={'RainTomorrow': 'target'}, inplace=True)
    logger.debug("Data preprocessing completed")
    return df

def save_data(train_data, test_data, data_path):
    raw_data_path = os.path.join(data_path, "raw")
    os.makedirs(raw_data_path, exist_ok=True)
    train_data.to_csv(os.path.join(raw_data_path, "train.csv"), index=False)
    test_data.to_csv(os.path.join(raw_data_path, "test.csv"), index=False)
    logger.debug("Train and test data saved to %s", raw_data_path)

# =========================
# Main
# =========================
def main():
    try:
        params = load_params("data_ingestion")
        test_size = params.get("test_size", 0.2)
        random_state = params.get("random_state", 2)
        data_url = params.get("data_url")

        df = load_data(data_url)
        final_df = preprocess_data(df)

        train_data, test_data = train_test_split(
            final_df,
            test_size=test_size,
            random_state=random_state
        )

        save_data(train_data, test_data, data_path='./data')
        logger.debug("Data ingestion completed successfully")
    except Exception as e:
        logger.error("Data ingestion failed: %s", e)
        print(f"Error: {e}")

if __name__ == "__main__":
    main()
