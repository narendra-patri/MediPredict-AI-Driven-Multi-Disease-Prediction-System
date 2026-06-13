# SmartMed: AI-Powered Multi-Disease Diagnostic Tool

An end-to-end machine learning web app that predicts 5 diseases from patient input data using trained ML models served via a Streamlit dashboard.

---

## Diseases Covered
| # | Disease | Dataset | Model |
|---|---------|---------|-------|
| 1 | Diabetes | Pima Indians Diabetes (UCI) | Random Forest |
| 2 | Heart Disease | Heart Failure Prediction (Kaggle) | Random Forest |
| 3 | Parkinson's Disease | Parkinson's Dataset (UCI) | SVM |
| 4 | Chronic Kidney Disease | CKD Dataset (UCI) | Random Forest |
| 5 | Liver Disease | Indian Liver Patient Dataset (UCI) | XGBoost |

---

## Dataset Download Links
- **Diabetes**: https://www.kaggle.com/datasets/uciml/pima-indians-diabetes-database
- **Heart Disease**: https://www.kaggle.com/datasets/fedesoriano/heart-failure-prediction
- **Parkinson's**: https://www.kaggle.com/datasets/vikasukani/parkinsons-disease-data-set
- **Kidney Disease**: https://www.kaggle.com/datasets/mansoordaku/ckdisease
- **Liver Disease**: https://www.kaggle.com/datasets/uciml/indian-liver-patient-records

Download each CSV and place in the `datasets/` folder.

---

## Project Structure
```
SmartMed/
├── datasets/
│   ├── diabetes.csv
│   ├── heart.csv
│   ├── parkinsons.csv
│   ├── kidney_disease.csv
│   └── liver_disease.csv
├── notebooks/
│   ├── 01_diabetes_model.ipynb
│   ├── 02_heart_model.ipynb
│   ├── 03_parkinsons_model.ipynb
│   ├── 04_kidney_model.ipynb
│   └── 05_liver_model.ipynb
├── models/
│   ├── diabetes_model.pkl
│   ├── heart_model.pkl
│   ├── parkinsons_model.pkl
│   ├── kidney_model.pkl
│   └── liver_model.pkl
├── app/
│   └── app.py
├── requirements.txt
└── README.md
```

---

## Setup Instructions

### 1. Install dependencies
```bash
pip install -r requirements.txt
```

### 2. Download datasets
Place all 5 CSV files into the `datasets/` folder (see links above).

### 3. Train all models
Run all the .ipynb files in notebook folder

This saves all 5 `.pkl` model files into `models/`.

### 4. Run the Streamlit app
```bash
streamlit run app/app.py
```

---

## Tech Stack
- **Python 3.10+**
- **Scikit-learn, XGBoost** — ML models
- **Pandas, NumPy** — data processing
- **Streamlit** — web app
- **SHAP** — model explainability
- **Matplotlib, Seaborn** — visualization

---

## Results Summary
| Disease | Model | Accuracy |
|---------|-------|----------|
| Diabetes | Random Forest | ~80% |
| Heart Disease | Random Forest | ~87% |
| Parkinson's | SVM | ~93% |
| Chronic Kidney Disease | Random Forest | ~98% |
| Liver Disease | XGBoost | ~76% |
