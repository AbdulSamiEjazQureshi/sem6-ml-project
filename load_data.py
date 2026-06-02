import zipfile
from pathlib import Path

import pandas as pd


CSV_PATH = Path("data/predictive_maintenance.csv")
ZIP_PATH = Path("data/ai4i+2020+predictive+maintenance+dataset.zip")


def extract_dataset_if_needed():
    if CSV_PATH.exists():
        return

    if not ZIP_PATH.exists():
        raise FileNotFoundError("Dataset CSV and ZIP file were not found in data/")

    with zipfile.ZipFile(ZIP_PATH, "r") as zip_file:
        zip_file.extractall("data")

    extracted_file = Path("data/ai4i2020.csv")
    if extracted_file.exists():
        extracted_file.rename(CSV_PATH)


def load_dataset():
    extract_dataset_if_needed()
    data = pd.read_csv(CSV_PATH)
    data.columns = data.columns.str.replace("\ufeff", "").str.strip()
    return data


def show_dataset_summary():
    data = load_dataset()

    print("Dataset loaded successfully")
    print("Rows and columns:", data.shape)
    print("\nColumns:")
    print(list(data.columns))
    print("\nFailure count:")
    print(data["Target"].value_counts())
    print("\nMissing values:")
    print(data.isnull().sum())


if __name__ == "__main__":
    show_dataset_summary()
