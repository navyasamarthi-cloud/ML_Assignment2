# ML Assignment 2 - Final Submission Template (Single PDF)

## Cover Details
- Course: Machine Learning
- Assignment: Assignment 2
- Name: Navya Alugubelli
- ID: 2025ab05030
- Date: 15/02/2026

---

## 1) GitHub Repository Link
Paste your repository URL:

https://github.com/navyasamarthi-cloud/ML_Assignment2

Repository should contain:
- app.py
- train_models.py
- __init__.py
- requirements.txt
- README.md

---

## 2) Live Streamlit App Link
Paste your deployed Streamlit URL:

https://mlassignment2-djjjjz6cnrw9rhhi6t4s4b.streamlit.app/

Expected when opened:
- Interactive frontend loads successfully
- Model dropdown works
- CSV upload works
- Metrics/confusion matrix/classification report visible

---

## 3) Screenshot of BITS Virtual Lab Execution
Insert one screenshot here showing assignment execution on BITS Virtual Lab.

[INSERT_SCREENSHOT_HERE]

---

## 4) README Content (Include This Exactly in PDF)

### a. Problem statement
Build and compare six classification models on one dataset, evaluate them using required metrics, and deploy an interactive Streamlit app with model selection, CSV test upload, and visual performance outputs.

### b. Dataset description
- Dataset Name: Breast Cancer Wisconsin (Diagnostic)
- Source: UCI Machine Learning Repository (also available through scikit-learn)
- Task Type: Binary Classification
- Instances: 569 (>= 500)
- Features: 30 numeric features (>= 12)
- Target: target (0 = malignant, 1 = benign)

### c. Models used
1. Logistic Regression
2. Decision Tree Classifier
3. K-Nearest Neighbors (KNN)
4. Naive Bayes (GaussianNB)
5. Random Forest (Ensemble)
6. XGBoost (Ensemble)

### Comparison Table

| ML Model Name | Accuracy | AUC | Precision | Recall | F1 | MCC |
|---|---:|---:|---:|---:|---:|---:|
| Logistic Regression | 0.9825 | 0.9954 | 0.9861 | 0.9861 | 0.9861 | 0.9623 |
| Decision Tree | 0.9211 | 0.9163 | 0.9565 | 0.9167 | 0.9362 | 0.8341 |
| KNN | 0.9737 | 0.9884 | 0.9600 | 1.0000 | 0.9796 | 0.9442 |
| Naive Bayes | 0.9298 | 0.9868 | 0.9444 | 0.9444 | 0.9444 | 0.8492 |
| Random Forest (Ensemble) | 0.9474 | 0.9940 | 0.9583 | 0.9583 | 0.9583 | 0.8869 |
| XGBoost (Ensemble) | 0.9561 | 0.9954 | 0.9467 | 0.9861 | 0.9660 | 0.9058 |

### Observations Table

| ML Model Name | Observation about model performance |
|---|---|
| Logistic Regression | Best overall in this run (highest Accuracy/F1/MCC), indicating strong linear separability with robust generalization. |
| Decision Tree | Lowest metrics among all models; likely affected by single-tree variance despite depth control. |
| KNN | Very strong Recall (1.0000) and high F1; performs well after scaling and captures local neighborhoods effectively. |
| Naive Bayes | Competitive AUC but lower Accuracy/MCC than top models; conditional independence assumption limits final class decisions. |
| Random Forest (Ensemble) | Strong and stable performance with high AUC; better generalization than a single decision tree. |
| XGBoost (Ensemble) | Excellent AUC and Recall with strong overall scores; close to top performance and effective on tabular features. |

---

## 5) Final Checklist Before Upload
- Single PDF prepared
- GitHub link included and clickable
- Streamlit live link included and clickable
- BITS Virtual Lab screenshot inserted
- README content included in PDF
- Submission done only once on Taxila
