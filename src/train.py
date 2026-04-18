from pathlib import Path

import matplotlib.pyplot as plt
import mlflow
import mlflow.sklearn
from imblearn.over_sampling import SMOTE
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.preprocessing import OneHotEncoder

from data_preparation import (
    clean_data,
    load_data,
    split_features_target,
    split_train_test,
)

RANDOM_STATE = 42
EXPERIMENT_NAME = "Mushroom Classification"


def main():
    project_root = Path(__file__).resolve().parents[1]
    data_path = project_root / "data" / "secondary_data.csv"

    mlflow.set_tracking_uri("http://127.0.0.1:5000")
    mlflow.set_experiment(EXPERIMENT_NAME)

    df = load_data(data_path)
    df = clean_data(df)
    X, y = split_features_target(df)
    X_train, X_test, y_train, y_test = split_train_test(X, y)

    categorical_cols = X.columns.tolist()

    preprocessor = ColumnTransformer(
        transformers=[
            ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_cols)
        ]
    )

    X_train_encoded = preprocessor.fit_transform(X_train)
    X_test_encoded = preprocessor.transform(X_test)

    smote = SMOTE(random_state=RANDOM_STATE)
    X_train_resampled, y_train_resampled = smote.fit_resample(X_train_encoded, y_train)

    model = RandomForestClassifier(random_state=RANDOM_STATE, n_estimators=200)

    with mlflow.start_run(run_name="random_forest_final"):
        model.fit(X_train_resampled, y_train_resampled)
        y_pred = model.predict(X_test_encoded)

        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred, pos_label=1)
        recall = recall_score(y_test, y_pred, pos_label=1)
        f1 = f1_score(y_test, y_pred, pos_label=1)

        print("\n=== Classification Report ===")
        print(classification_report(y_test, y_pred, target_names=["Edible (0)", "Poisonous (1)"]))
        print(f"F1-score (Poisonous = 1): {f1:.4f}")

        mlflow.log_param("model_name", "RandomForestClassifier")
        mlflow.log_param("random_state", RANDOM_STATE)
        mlflow.log_param("n_estimators", 200)
        mlflow.log_param("use_smote", True)

        mlflow.log_metric("accuracy", accuracy)
        mlflow.log_metric("precision_poisonous", precision)
        mlflow.log_metric("recall_poisonous", recall)
        mlflow.log_metric("f1_poisonous", f1)

        cm = confusion_matrix(y_test, y_pred)
        disp = ConfusionMatrixDisplay(
            confusion_matrix=cm,
            display_labels=["Edible (0)", "Poisonous (1)"]
        )
        disp.plot(cmap="Blues", values_format="d")
        plt.title("Confusion Matrix - Random Forest")

        artifact_dir = project_root / "artifacts"
        artifact_dir.mkdir(exist_ok=True)

        cm_path = artifact_dir / "confusion_matrix_rf.png"
        plt.savefig(cm_path, bbox_inches="tight")
        plt.close()

        mlflow.log_artifact(str(cm_path))
        mlflow.sklearn.log_model(model, name="random_forest_model")


if __name__ == "__main__":
    main()