from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Tuple

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.datasets import load_breast_cancer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    matthews_corrcoef,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import GradientBoostingClassifier

try:
    from xgboost import XGBClassifier
    HAS_XGBOOST = True
except ModuleNotFoundError:
    XGBClassifier = None
    HAS_XGBOOST = False


BASE_DIR = Path(__file__).resolve().parent
ARTIFACT_DIR = BASE_DIR / "artifacts"
FEATURES_FILE = ARTIFACT_DIR / "feature_columns.json"
METRICS_FILE = ARTIFACT_DIR / "metrics.csv"
CONF_MAT_FILE = ARTIFACT_DIR / "confusion_matrices.json"
CLASS_REPORT_FILE = ARTIFACT_DIR / "classification_reports.json"
TEST_SET_FILE = ARTIFACT_DIR / "test_set.csv"

MODEL_FILENAMES = {
    "Logistic Regression": "logistic_regression.pkl",
    "Decision Tree": "decision_tree.pkl",
    "KNN": "knn.pkl",
    "Naive Bayes": "naive_bayes.pkl",
    "Random Forest": "random_forest.pkl",
    "XGBoost": "xgboost.pkl",
}



def _get_dataset() -> Tuple[pd.DataFrame, pd.Series]:
    data = load_breast_cancer(as_frame=True)
    features = data.frame.drop(columns=["target"])
    target = data.frame["target"]
    return features, target



def _build_preprocessor(numeric_features: list[str]) -> ColumnTransformer:
    numeric_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )

    return ColumnTransformer(
        transformers=[("num", numeric_transformer, numeric_features)],
        remainder="drop",
    )



def _build_models(feature_names: list[str]) -> Dict[str, Pipeline]:
    preprocessor = _build_preprocessor(feature_names)

    models: Dict[str, Pipeline] = {
        "Logistic Regression": Pipeline(
            steps=[
                ("preprocessor", preprocessor),
                ("classifier", LogisticRegression(max_iter=2000, random_state=42)),
            ]
        ),
        "Decision Tree": Pipeline(
            steps=[
                ("preprocessor", preprocessor),
                ("classifier", DecisionTreeClassifier(max_depth=5, random_state=42)),
            ]
        ),
        "KNN": Pipeline(
            steps=[
                ("preprocessor", preprocessor),
                ("classifier", KNeighborsClassifier(n_neighbors=7)),
            ]
        ),
        "Naive Bayes": Pipeline(
            steps=[
                ("preprocessor", preprocessor),
                ("classifier", GaussianNB()),
            ]
        ),
        "Random Forest": Pipeline(
            steps=[
                ("preprocessor", preprocessor),
                (
                    "classifier",
                    RandomForestClassifier(
                        n_estimators=300,
                        max_depth=8,
                        random_state=42,
                    ),
                ),
            ]
        ),
        "XGBoost": Pipeline(
            steps=[
                ("preprocessor", preprocessor),
                (
                    "classifier",
                    XGBClassifier(
                        n_estimators=250,
                        learning_rate=0.05,
                        max_depth=4,
                        subsample=0.9,
                        colsample_bytree=0.9,
                        random_state=42,
                        eval_metric="logloss",
                    )
                    if HAS_XGBOOST
                    else GradientBoostingClassifier(random_state=42),
                ),
            ]
        ),
    }

    return models



def _evaluate_model(model: Pipeline, x_test: pd.DataFrame, y_test: pd.Series) -> dict:
    y_pred = model.predict(x_test)

    if hasattr(model, "predict_proba"):
        y_prob = model.predict_proba(x_test)[:, 1]
    else:
        y_prob = y_pred

    metrics = {
        "Accuracy": accuracy_score(y_test, y_pred),
        "AUC": roc_auc_score(y_test, y_prob),
        "Precision": precision_score(y_test, y_pred, zero_division=0),
        "Recall": recall_score(y_test, y_pred, zero_division=0),
        "F1": f1_score(y_test, y_pred, zero_division=0),
        "MCC": matthews_corrcoef(y_test, y_pred),
        "Confusion Matrix": confusion_matrix(y_test, y_pred).tolist(),
        "Classification Report": classification_report(
            y_test, y_pred, output_dict=True, zero_division=0
        ),
    }

    return metrics



def _save_json(path: Path, payload: dict) -> None:
    with path.open("w", encoding="utf-8") as file:
        json.dump(payload, file, indent=2)



def train_and_save_models(force_retrain: bool = False) -> pd.DataFrame:
    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)

    if METRICS_FILE.exists() and not force_retrain:
        return pd.read_csv(METRICS_FILE)

    x, y = _get_dataset()
    feature_names = list(x.columns)

    x_train, x_test, y_train, y_test = train_test_split(
        x,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y,
    )

    models = _build_models(feature_names)

    all_metrics = []
    confusion_matrices: dict[str, list[list[int]]] = {}
    classification_reports: dict[str, dict] = {}

    for model_name, model in models.items():
        model.fit(x_train, y_train)

        metrics = _evaluate_model(model, x_test, y_test)

        confusion_matrices[model_name] = metrics.pop("Confusion Matrix")
        classification_reports[model_name] = metrics.pop("Classification Report")

        model_path = ARTIFACT_DIR / MODEL_FILENAMES[model_name]
        joblib.dump(model, model_path)

        all_metrics.append({"ML Model Name": model_name, **metrics})

    metrics_df = pd.DataFrame(all_metrics)

    metrics_df.to_csv(METRICS_FILE, index=False)
    _save_json(CONF_MAT_FILE, confusion_matrices)
    _save_json(CLASS_REPORT_FILE, classification_reports)

    _save_json(FEATURES_FILE, {"feature_columns": feature_names, "target_column": "target"})

    test_with_target = x_test.copy()
    test_with_target["target"] = y_test.values
    test_with_target.to_csv(TEST_SET_FILE, index=False)

    return metrics_df



def load_models() -> Dict[str, Pipeline]:
    loaded_models: Dict[str, Pipeline] = {}

    for model_name, filename in MODEL_FILENAMES.items():
        model_path = ARTIFACT_DIR / filename
        if model_path.exists():
            loaded_models[model_name] = joblib.load(model_path)

    return loaded_models



def ensure_artifacts() -> pd.DataFrame:
    if not METRICS_FILE.exists():
        return train_and_save_models(force_retrain=True)
    return pd.read_csv(METRICS_FILE)



def load_feature_info() -> dict:
    with FEATURES_FILE.open("r", encoding="utf-8") as file:
        return json.load(file)



def load_confusion_matrices() -> dict:
    with CONF_MAT_FILE.open("r", encoding="utf-8") as file:
        return json.load(file)



def load_classification_reports() -> dict:
    with CLASS_REPORT_FILE.open("r", encoding="utf-8") as file:
        return json.load(file)
