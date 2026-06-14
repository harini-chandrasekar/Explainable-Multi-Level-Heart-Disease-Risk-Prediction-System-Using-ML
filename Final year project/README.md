# 🫀 An Explainable Multi-Level Heart Disease Risk Prediction System

**Authors:** Harini C (510622205031), Amruta B S (510622205011)  
**Department:** Information Technology, CAHCET

---

## Project Structure

```
heart-risk-prediction/
├── heart_risk_predictor.html       # ✅ Full web app (open in browser, no install needed)
├── heart_disease_ml_backend.py     # 🐍 Python ML pipeline + Flask API
├── README.md                       # This file
└── outputs/                        # Generated after running backend
    ├── models.pkl
    ├── evaluation_plots.png
    └── feature_importance.png
```

---

## 🚀 Quick Start — Web App (No Installation)

Just open `heart_risk_predictor.html` in any modern browser.

**Features:**
- Multi-level risk prediction: No Risk / Low Risk / Medium Risk / High Risk
- 6 ML model comparison (XGBoost, RF, SVM, LR, KNN, Gradient Boosting)
- Feature importance visualization (XAI)
- AI-powered clinical explanation via Claude API
- Lifestyle factor inputs (smoking, activity, stress, sleep)
- Derived features: BMI, pulse pressure, age groups

---

## 🐍 Python ML Backend

### Install Dependencies
```bash
pip install pandas numpy scikit-learn xgboost imbalanced-learn \
            matplotlib seaborn flask flask-cors shap
```

### Run the Full Pipeline
```bash
python heart_disease_ml_backend.py
```

This will:
1. Load the UCI Heart Disease dataset (or generate synthetic data)
2. Apply feature engineering (BMI, pulse pressure, age groups, HR reserve)
3. Handle class imbalance with **SMOTE**
4. Train 6 models with **GridSearchCV + 5-fold cross-validation**
5. Evaluate using Accuracy, ROC-AUC, Confusion Matrix
6. Save plots to `outputs/`
7. Start Flask REST API on `http://localhost:5000`

### API Endpoints

**POST `/predict`**
```json
{
  "age": 52, "sex": 1, "cp": 3, "trestbps": 145,
  "chol": 270, "fbs": 1, "restecg": 0, "thalach": 148,
  "exang": 1, "oldpeak": 2.1, "slope": 2, "ca": 2, "thal": 3
}
```

**Response:**
```json
{
  "risk_level": "High Risk",
  "risk_probability": 78.4,
  "best_model": "XGBoost",
  "all_model_scores": {
    "XGBoost": 78.4, "Random Forest": 76.1, "SVM": 72.3,
    "Logistic Regression": 69.8, "KNN": 71.2, "Gradient Boosting": 77.6
  },
  "feature_importances": { "ca": 0.22, "thal": 0.17, ... }
}
```

---

## 🧠 System Architecture

```
Dataset (UCI / Kaggle)
    ↓
Feature Engineering (BMI, Pulse Pressure, Age Groups, HR Reserve)
    ↓
Imbalanced Data Handling (SMOTE / ADASYN)
    ↓
Model Training (LR / KNN / SVM / RF / GB / XGBoost)
    ↓
GridSearchCV Hyperparameter Tuning (5-fold CV)
    ↓
Auto Model Selection (Best ROC-AUC)
    ↓
Multi-Level Risk Classification
  No Risk (<20%) | Low Risk (20-45%) | Medium Risk (45-70%) | High Risk (>70%)
    ↓
Explainable AI (Feature Importance + Probability Scores + Claude LLM)
    ↓
Real-Time Web Interface (Flask API + HTML/JS Frontend)
```

---

## 📊 Implemented ML Models

| Model | Notes |
|---|---|
| Logistic Regression | Baseline, class_weight='balanced' |
| K-Nearest Neighbors | Tuned k and distance weighting |
| Support Vector Machine | RBF kernel, probability calibration |
| Random Forest | 100-200 estimators, max_depth tuned |
| Gradient Boosting | sklearn GBM implementation |
| XGBoost | Scale_pos_weight for imbalance, best performer |

---

## 📈 Evaluation Metrics

- **Accuracy** — Overall correct predictions
- **ROC-AUC** — Discrimination ability across thresholds
- **Precision / Recall / F1** — Per-class performance
- **Confusion Matrix** — Visual breakdown of TP/TN/FP/FN

---

## 🔬 Explainability (XAI)

- **Feature Importance** from tree-based models
- **Probability Scores** for each risk class
- **Claude LLM** generates natural-language clinical explanations
- Risk classified into 4 levels (not just binary Yes/No)

---

## ⚠️ Disclaimer

This system is for **educational and research purposes only**. It does not constitute medical advice. Always consult a qualified cardiologist for clinical decisions.
