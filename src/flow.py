from pathlib import Path

from prefect import flow, task

from data_preparation import (
    clean_data,
    load_data,
    split_features_target,
    split_train_test,
)
from train import main as train_model


@task
def task_load_data(data_path):
    return load_data(data_path)


@task
def task_clean_data(df):
    return clean_data(df)


@task
def task_split_features(df):
    return split_features_target(df)


@task
def task_split_train_test(X, y):
    return split_train_test(X, y)


@task
def task_train_model():
    train_model()


@flow(name="ml-pipeline")
def ml_pipeline(data_path):
    df = task_load_data(data_path)
    df_clean = task_clean_data(df)
    X, y = task_split_features(df_clean)
    X_train, X_test, y_train, y_test = task_split_train_test(X, y)

    print("Pipeline ejecutado correctamente")
    print("Train:", X_train.shape, y_train.shape)
    print("Test:", X_test.shape, y_test.shape)

    task_train_model()


if __name__ == "__main__":
    project_root = Path(__file__).resolve().parents[1]
    data_path = project_root / "data" / "secondary_data.csv"
    ml_pipeline(str(data_path))