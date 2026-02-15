from __future__ import annotations

import numpy as np
import pandas as pd
import seaborn as sns
import streamlit as st
import matplotlib.pyplot as plt
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    matthews_corrcoef,
    precision_score,
    recall_score,
    roc_auc_score,
    classification_report,
)

from train_models import (
    ensure_artifacts,
    load_classification_reports,
    load_confusion_matrices,
    load_feature_info,
    load_models,
    train_and_save_models,
)


st.set_page_config(page_title="ML Assignment 2 - Classification", layout="wide")
st.title("Classification Model Comparison and Demo")
st.caption("BITS ML Assignment 2: Logistic Regression, Decision Tree, KNN, Naive Bayes, Random Forest, XGBoost")

with st.sidebar:
    st.header("Controls")
    retrain = st.button("Retrain and refresh artifacts")

if retrain:
    metrics_df = train_and_save_models(force_retrain=True)
    st.success("Models retrained and artifacts updated.")
else:
    metrics_df = ensure_artifacts()

models = load_models()
feature_info = load_feature_info()
confusion_matrices = load_confusion_matrices()
classification_reports = load_classification_reports()

feature_columns = feature_info["feature_columns"]
target_column = feature_info["target_column"]

if not models:
    st.error("No saved models found. Click 'Retrain and refresh artifacts' in sidebar.")
    st.stop()

selected_model_name = st.selectbox("Select Model", options=list(models.keys()))
selected_model = models[selected_model_name]

st.subheader("Evaluation Metrics (Holdout Test Set)")
st.dataframe(metrics_df, use_container_width=True)

selected_row = metrics_df[metrics_df["ML Model Name"] == selected_model_name].iloc[0]
metric_cols = st.columns(6)
metric_cols[0].metric("Accuracy", f"{selected_row['Accuracy']:.4f}")
metric_cols[1].metric("AUC", f"{selected_row['AUC']:.4f}")
metric_cols[2].metric("Precision", f"{selected_row['Precision']:.4f}")
metric_cols[3].metric("Recall", f"{selected_row['Recall']:.4f}")
metric_cols[4].metric("F1", f"{selected_row['F1']:.4f}")
metric_cols[5].metric("MCC", f"{selected_row['MCC']:.4f}")

st.subheader("Confusion Matrix (Holdout Test Set)")
cm = np.array(confusion_matrices[selected_model_name])
fig, ax = plt.subplots(figsize=(4, 3))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", cbar=False, ax=ax)
ax.set_xlabel("Predicted")
ax.set_ylabel("Actual")
st.pyplot(fig)

st.subheader("Classification Report (Holdout Test Set)")
report_dict = classification_reports[selected_model_name]
report_df = pd.DataFrame(report_dict).transpose()
st.dataframe(report_df, use_container_width=True)

st.markdown("---")
st.subheader("Dataset Upload (CSV)")
st.write("Upload CSV test data. Include all required feature columns. If target column is present, metrics are computed on uploaded data.")

uploaded_file = st.file_uploader("Upload test CSV", type=["csv"])

if uploaded_file is not None:
    input_df = pd.read_csv(uploaded_file)

    missing_features = [col for col in feature_columns if col not in input_df.columns]
    if missing_features:
        st.error(f"Missing feature columns: {missing_features}")
        st.stop()

    x_input = input_df[feature_columns]
    preds = selected_model.predict(x_input)

    result_df = input_df.copy()
    result_df["prediction"] = preds

    if hasattr(selected_model, "predict_proba"):
        result_df["prediction_probability"] = selected_model.predict_proba(x_input)[:, 1]

    st.subheader("Predictions")
    st.dataframe(result_df.head(100), use_container_width=True)

    csv_bytes = result_df.to_csv(index=False).encode("utf-8")
    st.download_button(
        label="Download predictions CSV",
        data=csv_bytes,
        file_name=f"predictions_{selected_model_name.replace(' ', '_').lower()}.csv",
        mime="text/csv",
    )

    if target_column in input_df.columns:
        y_true = input_df[target_column]
        y_pred = preds

        if hasattr(selected_model, "predict_proba"):
            y_prob = selected_model.predict_proba(x_input)[:, 1]
        else:
            y_prob = y_pred

        upload_metrics = {
            "Accuracy": accuracy_score(y_true, y_pred),
            "AUC": roc_auc_score(y_true, y_prob),
            "Precision": precision_score(y_true, y_pred, zero_division=0),
            "Recall": recall_score(y_true, y_pred, zero_division=0),
            "F1": f1_score(y_true, y_pred, zero_division=0),
            "MCC": matthews_corrcoef(y_true, y_pred),
        }

        st.subheader("Uploaded Data Metrics")
        metric_cols_upload = st.columns(6)
        metric_cols_upload[0].metric("Accuracy", f"{upload_metrics['Accuracy']:.4f}")
        metric_cols_upload[1].metric("AUC", f"{upload_metrics['AUC']:.4f}")
        metric_cols_upload[2].metric("Precision", f"{upload_metrics['Precision']:.4f}")
        metric_cols_upload[3].metric("Recall", f"{upload_metrics['Recall']:.4f}")
        metric_cols_upload[4].metric("F1", f"{upload_metrics['F1']:.4f}")
        metric_cols_upload[5].metric("MCC", f"{upload_metrics['MCC']:.4f}")

        st.subheader("Uploaded Data Confusion Matrix")
        upload_cm = confusion_matrix(y_true, y_pred)
        fig2, ax2 = plt.subplots(figsize=(4, 3))
        sns.heatmap(upload_cm, annot=True, fmt="d", cmap="Greens", cbar=False, ax=ax2)
        ax2.set_xlabel("Predicted")
        ax2.set_ylabel("Actual")
        st.pyplot(fig2)

        st.subheader("Uploaded Data Classification Report")
        upload_report = classification_report(y_true, y_pred, output_dict=True, zero_division=0)
        st.dataframe(pd.DataFrame(upload_report).transpose(), use_container_width=True)
