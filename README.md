# ML Assignment 2 - Classification Models + Streamlit Deployment

## a. Problem statement
Build and compare six classification models on one dataset, evaluate them using required metrics, and deploy an interactive Streamlit app with model selection, CSV test upload, and visual performance outputs.

## b. Dataset description
- **Dataset Name**: Breast Cancer Wisconsin (Diagnostic)
- **Source**: UCI Machine Learning Repository (also available through `scikit-learn`)
- **Task Type**: Binary Classification
- **Instances**: 569 (>= 500)
- **Features**: 30 numeric features (>= 12)
- **Target**: `target` (0 = malignant, 1 = benign)

## c. Models used
1. Logistic Regression
2. Decision Tree Classifier
3. K-Nearest Neighbors (KNN)
4. Naive Bayes (GaussianNB)
5. Random Forest (Ensemble)
6. XGBoost (Ensemble)

### Comparison table (fill this from generated `model/artifacts/metrics.csv` after training)

| ML Model Name | Accuracy | AUC | Precision | Recall | F1 | MCC |
|---|---:|---:|---:|---:|---:|---:|
| Logistic Regression | 0.9825 | 0.9954 | 0.9861 | 0.9861 | 0.9861 | 0.9623 |
| Decision Tree | 0.9211 | 0.9163 | 0.9565 | 0.9167 | 0.9362 | 0.8341 |
| KNN | 0.9737 | 0.9884 | 0.9600 | 1.0000 | 0.9796 | 0.9442 |
| Naive Bayes | 0.9298 | 0.9868 | 0.9444 | 0.9444 | 0.9444 | 0.8492 |
| Random Forest (Ensemble) | 0.9474 | 0.9940 | 0.9583 | 0.9583 | 0.9583 | 0.8869 |
| XGBoost (Ensemble) | 0.9561 | 0.9954 | 0.9467 | 0.9861 | 0.9660 | 0.9058 |

### Observations about each model

| ML Model Name | Observation about model performance |
|---|---|
| Logistic Regression | Best overall in this run (highest Accuracy/F1/MCC), indicating strong linear separability with robust generalization. |
| Decision Tree | Lowest metrics among all models; likely affected by single-tree variance despite depth control. |
| KNN | Very strong Recall (1.0000) and high F1; performs well after scaling and captures local neighborhoods effectively. |
| Naive Bayes | Competitive AUC but lower Accuracy/MCC than top models; conditional independence assumption limits final class decisions. |
| Random Forest (Ensemble) | Strong and stable performance with high AUC; better generalization than a single decision tree. |
| XGBoost (Ensemble) | Excellent AUC and Recall with strong overall scores; close to top performance and effective on tabular features. |

---

## Project structure

```text
project-folder/
│-- app.py
│-- requirements.txt
│-- README.md
│-- model/
│   │-- __init__.py
│   │-- train_models.py
│   │-- artifacts/   (generated after first run)
```

## Local run instructions

1. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
2. Start Streamlit:
   ```bash
   streamlit run app.py
   ```
3. On first run, models and artifacts are generated in `model/artifacts/`.

## Streamlit app features implemented
- CSV dataset upload option for test data
- Model selection dropdown
- Display of evaluation metrics (Accuracy, AUC, Precision, Recall, F1, MCC)
- Confusion matrix and classification report

## Deployment steps (Streamlit Community Cloud)
1. Push this project to a public GitHub repository.
2. Go to https://streamlit.io/cloud and sign in with GitHub.
3. Click **New App**.
4. Choose repository and branch.
5. Set main file path as `app.py`.
6. Deploy and copy the generated live app URL.

## Mandatory submission checklist (single PDF)
1. GitHub repository link (source code + `requirements.txt` + `README.md`)
2. Live Streamlit app link
3. Screenshot of assignment execution on BITS Virtual Lab
4. README content included in the PDF
