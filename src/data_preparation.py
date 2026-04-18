from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split


RANDOM_STATE = 42
TEST_SIZE = 0.2
TARGET_COL = "class"


def load_data(data_path: str | Path) -> pd.DataFrame:
    """
    Load the dataset from a CSV file using ';' as separator.
    """
    data_path = Path(data_path)
    df = pd.read_csv(data_path, sep=";")
    return df


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Clean the dataset:
    - remove duplicates
    - keep '?' for now as a categorical value
    """
    df = df.copy()
    df = df.drop_duplicates()
    return df


def split_features_target(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    """
    Split the dataset into features (X) and target (y).
    Encode target: poisonous = 1, edible = 0
    """
    X = df.drop(columns=[TARGET_COL])
    y = df[TARGET_COL].map({"p": 1, "e": 0})
    return X, y


def split_train_test(
    X: pd.DataFrame, y: pd.Series
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """
    Perform train-test split with stratification.
    """
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=y,
    )
    return X_train, X_test, y_train, y_test


if __name__ == "__main__":
    project_root = Path(__file__).resolve().parents[1]
    data_path = project_root / "data" / "secondary_data.csv"

    df = load_data(data_path)
    print("Dataset original:", df.shape)

    df_clean = clean_data(df)
    print("Dataset limpio:", df_clean.shape)

    X, y = split_features_target(df_clean)
    print("Features:", X.shape)
    print("Target:", y.shape)

    X_train, X_test, y_train, y_test = split_train_test(X, y)
    print("Train:", X_train.shape, y_train.shape)
    print("Test:", X_test.shape, y_test.shape)