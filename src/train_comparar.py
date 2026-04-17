from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
from imblearn.over_sampling import SMOTE
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    classification_report,
    confusion_matrix,
    f1_score,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

from data_preparation import (
    clean_data,
    load_data,
    split_features_target,
    split_train_test,
)


RANDOM_STATE = 42


def build_pipeline(categorical_cols: list[str]) -> Pipeline:
    preprocessor = ColumnTransformer(
        transformers=[
            ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_cols)
        ]
    )

    pipeline = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("model", RandomForestClassifier(random_state=RANDOM_STATE, n_estimators=200)),
        ]
    )
    return pipeline


if __name__ == "__main__":
    project_root = Path(__file__).resolve().parents[1]
    data_path = project_root / "data" / "secondary_data.csv"

    df = load_data(data_path)
    df = clean_data(df)
    X, y = split_features_target(df)
    X_train, X_test, y_train, y_test = split_train_test(X, y)

    categorical_cols = X.columns.tolist()

    # Preprocesamiento antes de SMOTE
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
    model.fit(X_train_resampled, y_train_resampled)

    y_pred = model.predict(X_test_encoded)

    print("\n=== Classification Report ===")
    print(classification_report(y_test, y_pred, target_names=["Edible (0)", "Poisonous (1)"]))

    f1_poisonous = f1_score(y_test, y_pred, pos_label=1)
    print(f"F1-score (Poisonous = 1): {f1_poisonous:.4f}")

    cm = confusion_matrix(y_test, y_pred)
    disp = ConfusionMatrixDisplay(
        confusion_matrix=cm,
        display_labels=["Edible (0)", "Poisonous (1)"]
    )
    disp.plot(cmap="Blues", values_format="d")
    plt.title("Confusion Matrix - Random Forest")
    plt.show()