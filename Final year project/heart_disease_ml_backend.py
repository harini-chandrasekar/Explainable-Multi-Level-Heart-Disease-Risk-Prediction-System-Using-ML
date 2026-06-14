"""
╔══════════════════════════════════════════════════════════════════╗
║  An Explainable Multi-Level Heart Disease Risk Prediction System ║
║  Complete Python ML Backend                                      ║
║  Authors: Harini C (510622205031), Amruta B S (510622205011)    ║
║  Dept. of Information Technology, CAHCET                        ║
╚══════════════════════════════════════════════════════════════════╝

SETUP:
  pip install pandas numpy scikit-learn xgboost imbalanced-learn
              matplotlib seaborn flask shap

RUN SERVER:
  python heart_disease_ml_backend.py
"""

# ─── Imports ──────────────────────────────────────────────────────────────────
import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings('ignore')

from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import (accuracy_score, classification_report,
                             confusion_matrix, roc_auc_score, roc_curve)
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from xgboost import XGBClassifier
from imblearn.over_sampling import SMOTE, ADASYN
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
import json
import os

# ─── 1. DATASET LOADING & FEATURE ENGINEERING ────────────────────────────────

def load_uci_dataset():
    """
    Load UCI Heart Disease dataset.
    If not available, generate a realistic synthetic dataset.
    """
    try:
        url = ("https://archive.ics.uci.edu/ml/machine-learning-databases/"
               "heart-disease/processed.cleveland.data")
        cols = ['age','sex','cp','trestbps','chol','fbs','restecg',
                'thalach','exang','oldpeak','slope','ca','thal','target']
        df = pd.read_csv(url, names=cols, na_values='?')
        df.dropna(inplace=True)
        df['target'] = (df['target'] > 0).astype(int)
        print(f"[✓] UCI dataset loaded: {df.shape}")
        return df
    except Exception:
        print("[!] UCI dataset unavailable — generating synthetic dataset")
        return generate_synthetic_dataset()


def generate_synthetic_dataset(n=1200, seed=42):
    """
    Generate realistic synthetic heart disease data
    based on UCI Heart Disease feature distributions.
    """
    rng = np.random.default_rng(seed)
    n_pos = int(n * 0.46)   # ~46% positive (heart disease present)
    n_neg = n - n_pos

    def make_group(size, has_disease):
        hd = int(has_disease)
        age = rng.normal(56 + 3*hd, 9, size).clip(29, 77)
        return pd.DataFrame({
            'age':      age.round(),
            'sex':      rng.choice([0,1], size, p=[0.32,0.68]),
            'cp':       rng.choice([0,1,2,3], size, p=[0.12,0.16,0.28,0.44] if hd
                                   else [0.24,0.35,0.28,0.13]),
            'trestbps': rng.normal(134+8*hd, 18, size).clip(94, 200).round(),
            'chol':     rng.normal(251, 51, size).clip(126, 564).round(),
            'fbs':      rng.choice([0,1], size, p=[0.82-0.12*hd, 0.18+0.12*hd]),
            'restecg':  rng.choice([0,1,2], size, p=[0.52-0.1*hd,0.36+0.05*hd,0.12+0.05*hd]),
            'thalach':  rng.normal(149-20*hd, 23, size).clip(71, 202).round(),
            'exang':    rng.choice([0,1], size, p=[0.68-0.3*hd,0.32+0.3*hd]),
            'oldpeak':  np.abs(rng.normal(0.6+2.4*hd, 1.4, size)).clip(0, 6.2).round(1),
            'slope':    rng.choice([0,1,2], size, p=[0.12+0.1*hd,0.38+0.1*hd,0.5-0.2*hd]),
            'ca':       rng.choice([0,1,2,3], size, p=[0.58-0.22*hd,0.22+0.1*hd,0.13+0.08*hd,0.07+0.04*hd]),
            'thal':     rng.choice([1,2,3], size, p=[0.06,0.35-0.2*hd,0.59+0.2*hd] if hd
                                   else [0.06,0.55,0.39]),
            'target':   [hd]*size
        })

    df = pd.concat([make_group(n_pos,1), make_group(n_neg,0)], ignore_index=True)
    df = df.sample(frac=1, random_state=seed).reset_index(drop=True)
    print(f"[✓] Synthetic dataset generated: {df.shape}")
    return df


def engineer_features(df):
    """Add derived features: BMI placeholder, pulse pressure, age groups."""
    df = df.copy()

    # Pulse pressure proxy (systolic BP − estimated diastolic)
    df['pulse_pressure'] = (df['trestbps'] * 0.4).round(1)

    # Age group encoding
    df['age_group'] = pd.cut(df['age'], bins=[0,40,55,65,120],
                             labels=[0,1,2,3]).astype(int)

    # HR reserve proxy (rough estimate)
    df['hr_reserve'] = (220 - df['age'] - df['thalach']).clip(0, 160)

    # ST * angina interaction
    df['st_angina'] = df['oldpeak'] * df['exang']

    print(f"[✓] Feature engineering complete. Shape: {df.shape}")
    return df


# ─── 2. MULTI-LEVEL RISK LABEL ASSIGNMENT ────────────────────────────────────

def assign_risk_levels(df, model, scaler, feature_cols):
    """
    Assign 4-level risk labels using predicted probabilities:
      No Risk (<20%), Low Risk (20-45%), Medium Risk (45-70%), High Risk (>70%)
    """
    X = scaler.transform(df[feature_cols])
    probs = model.predict_proba(X)[:, 1]
    df['risk_prob'] = probs
    df['risk_level'] = pd.cut(probs,
                               bins=[-0.01, 0.20, 0.45, 0.70, 1.01],
                               labels=['No Risk','Low Risk','Medium Risk','High Risk'])
    return df


# ─── 3. MODEL TRAINING ────────────────────────────────────────────────────────

def train_models(X_train, y_train):
    """Train all 6 classifiers with GridSearchCV hyperparameter tuning."""

    models_config = {
        'Logistic Regression': {
            'model': LogisticRegression(max_iter=2000, class_weight='balanced'),
            'params': {'C': [0.01, 0.1, 1, 10], 'solver': ['lbfgs','liblinear']}
        },
        'KNN': {
            'model': KNeighborsClassifier(),
            'params': {'n_neighbors': [3,5,7,9,11], 'weights': ['uniform','distance']}
        },
        'SVM': {
            'model': SVC(probability=True, class_weight='balanced'),
            'params': {'C': [0.1,1,10], 'kernel': ['rbf','linear'], 'gamma': ['scale','auto']}
        },
        'Random Forest': {
            'model': RandomForestClassifier(class_weight='balanced', random_state=42),
            'params': {'n_estimators': [100,200], 'max_depth': [4,6,8],
                       'min_samples_split': [2,5]}
        },
        'Gradient Boosting': {
            'model': GradientBoostingClassifier(random_state=42),
            'params': {'n_estimators': [100,200], 'learning_rate': [0.05,0.1],
                       'max_depth': [3,4,5]}
        },
        'XGBoost': {
            'model': XGBClassifier(eval_metric='logloss', random_state=42,
                                   scale_pos_weight=(y_train==0).sum()/(y_train==1).sum()),
            'params': {'n_estimators': [100,200], 'learning_rate': [0.05,0.1],
                       'max_depth': [3,4,5], 'subsample': [0.8,1.0]}
        },
    }

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    trained = {}
    results = {}

    for name, cfg in models_config.items():
        print(f"  [→] Tuning {name}...")
        gs = GridSearchCV(cfg['model'], cfg['params'],
                          cv=cv, scoring='roc_auc', n_jobs=-1, verbose=0)
        gs.fit(X_train, y_train)
        trained[name] = gs.best_estimator_
        results[name] = {
            'best_params': gs.best_params_,
            'cv_roc_auc': round(gs.best_score_, 4)
        }
        print(f"     CV ROC-AUC: {gs.best_score_:.4f}  Params: {gs.best_params_}")

    return trained, results


# ─── 4. EVALUATION ────────────────────────────────────────────────────────────

def evaluate_models(trained_models, X_test, y_test, feature_names, out_dir='outputs'):
    """Compute metrics and generate evaluation plots."""
    os.makedirs(out_dir, exist_ok=True)
    metrics = {}

    for name, model in trained_models.items():
        y_pred  = model.predict(X_test)
        y_proba = model.predict_proba(X_test)[:,1]
        acc = accuracy_score(y_test, y_pred)
        auc = roc_auc_score(y_test, y_proba)
        metrics[name] = {'accuracy': round(acc,4), 'roc_auc': round(auc,4)}
        print(f"  {name:20s}  Acc={acc:.4f}  AUC={auc:.4f}")

    # ── ROC Curve Plot ──
    fig, axes = plt.subplots(1, 2, figsize=(14,5))
    fig.patch.set_facecolor('#0a0e1a')
    for ax in axes:
        ax.set_facecolor('#111827')
        ax.tick_params(colors='#6b8cae'); ax.xaxis.label.set_color('#6b8cae')
        ax.yaxis.label.set_color('#6b8cae'); ax.title.set_color('#e8f0fe')
        for spine in ax.spines.values(): spine.set_edgecolor('#1e2d42')

    colors = ['#3b82f6','#10b981','#f59e0b','#f97316','#ef4444','#a78bfa']
    for (name, model), color in zip(trained_models.items(), colors):
        fpr, tpr, _ = roc_curve(y_test, model.predict_proba(X_test)[:,1])
        auc = metrics[name]['roc_auc']
        axes[0].plot(fpr, tpr, label=f'{name} (AUC={auc})', color=color, lw=2)
    axes[0].plot([0,1],[0,1],'--',color='#6b8cae',lw=1)
    axes[0].set(xlabel='False Positive Rate', ylabel='True Positive Rate', title='ROC-AUC Curves')
    axes[0].legend(fontsize=8, facecolor='#111827', labelcolor='#e8f0fe')

    # ── Confusion Matrix for Best Model ──
    best_name = max(metrics, key=lambda k: metrics[k]['roc_auc'])
    cm = confusion_matrix(y_test, trained_models[best_name].predict(X_test))
    sns.heatmap(cm, annot=True, fmt='d', ax=axes[1],
                cmap='Blues', linewidths=0.5,
                annot_kws={'color':'white','size':14})
    axes[1].set(xlabel='Predicted', ylabel='Actual',
                title=f'Confusion Matrix — {best_name}')
    axes[1].set_xticklabels(['No Disease','Disease'], color='#6b8cae')
    axes[1].set_yticklabels(['No Disease','Disease'], color='#6b8cae', rotation=0)

    plt.tight_layout()
    plt.savefig(f'{out_dir}/evaluation_plots.png', dpi=150, bbox_inches='tight',
                facecolor='#0a0e1a')
    plt.close()
    print(f"[✓] Evaluation plots saved → {out_dir}/evaluation_plots.png")

    return metrics, best_name


def plot_feature_importance(best_model, best_name, feature_names, out_dir='outputs'):
    """Plot feature importance for tree-based models."""
    if not hasattr(best_model, 'feature_importances_'):
        print("[!] Feature importance not available for this model type")
        return

    importances = best_model.feature_importances_
    idx = np.argsort(importances)[::-1]

    fig, ax = plt.subplots(figsize=(10, 6))
    fig.patch.set_facecolor('#0a0e1a')
    ax.set_facecolor('#111827')
    for spine in ax.spines.values(): spine.set_edgecolor('#1e2d42')
    ax.tick_params(colors='#6b8cae')
    ax.xaxis.label.set_color('#6b8cae'); ax.yaxis.label.set_color('#6b8cae')
    ax.title.set_color('#e8f0fe')

    colors_bar = ['#3b82f6' if i==0 else '#1d4ed8' for i in range(len(idx))]
    bars = ax.barh([feature_names[i] for i in idx[::-1]],
                   importances[idx[::-1]], color=colors_bar[::-1], height=0.6)
    ax.set(xlabel='Importance Score', title=f'Feature Importances — {best_name}')

    plt.tight_layout()
    plt.savefig(f'{out_dir}/feature_importance.png', dpi=150, bbox_inches='tight',
                facecolor='#0a0e1a')
    plt.close()
    print(f"[✓] Feature importance saved → {out_dir}/feature_importance.png")


# ─── 5. FLASK API SERVER ──────────────────────────────────────────────────────

def create_flask_app(trained_models, best_model_name, scaler, feature_cols):
    """Create Flask REST API for real-time prediction."""
    try:
        from flask import Flask, request, jsonify
        from flask_cors import CORS
    except ImportError:
        print("[!] Flask not installed. Run: pip install flask flask-cors")
        return None

    app = Flask(__name__)
    CORS(app)

    RISK_THRESHOLDS = [
        (0.20, 'No Risk'),
        (0.45, 'Low Risk'),
        (0.70, 'Medium Risk'),
        (1.00, 'High Risk'),
    ]

    @app.route('/health', methods=['GET'])
    def health():
        return jsonify({'status': 'ok', 'best_model': best_model_name})

    @app.route('/predict', methods=['POST'])
    def predict():
        try:
            data = request.json

            # Build feature vector
            row = pd.DataFrame([{
                'age':      float(data['age']),
                'sex':      float(data['sex']),
                'cp':       float(data['cp']),
                'trestbps': float(data['trestbps']),
                'chol':     float(data['chol']),
                'fbs':      float(data['fbs']),
                'restecg':  float(data['restecg']),
                'thalach':  float(data['thalach']),
                'exang':    float(data['exang']),
                'oldpeak':  float(data['oldpeak']),
                'slope':    float(data['slope']),
                'ca':       float(data['ca']),
                'thal':     float(data['thal']),
                # Derived
                'pulse_pressure': float(data['trestbps']) * 0.4,
                'age_group': 0 if float(data['age'])<40 else (1 if float(data['age'])<55 else (2 if float(data['age'])<65 else 3)),
                'hr_reserve': max(0, 220 - float(data['age']) - float(data['thalach'])),
                'st_angina':  float(data['oldpeak']) * float(data['exang']),
            }])

            # Scale
            X = scaler.transform(row[feature_cols])

            # All model predictions
            model_preds = {}
            for name, model in trained_models.items():
                prob = float(model.predict_proba(X)[0, 1])
                model_preds[name] = round(prob * 100, 1)

            # Best model ensemble
            best_prob = trained_models[best_model_name].predict_proba(X)[0, 1]
            risk_label = next(lbl for thresh, lbl in RISK_THRESHOLDS if best_prob <= thresh)

            # Feature importances (if available)
            feat_imp = {}
            bm = trained_models[best_model_name]
            if hasattr(bm, 'feature_importances_'):
                feat_imp = dict(zip(feature_cols, bm.feature_importances_.tolist()))

            return jsonify({
                'success': True,
                'risk_level': risk_label,
                'risk_probability': round(best_prob * 100, 1),
                'best_model': best_model_name,
                'all_model_scores': model_preds,
                'feature_importances': feat_imp,
            })

        except Exception as e:
            return jsonify({'success': False, 'error': str(e)}), 400

    return app


# ─── 6. MAIN PIPELINE ─────────────────────────────────────────────────────────

def main():
    print("\n" + "="*65)
    print("  Multi-Level Heart Disease Risk Prediction System")
    print("  CAHCET — Department of Information Technology")
    print("="*65 + "\n")

    # ── Load & Engineer ──
    df = load_uci_dataset()
    df = engineer_features(df)

    feature_cols = ['age','sex','cp','trestbps','chol','fbs','restecg',
                    'thalach','exang','oldpeak','slope','ca','thal',
                    'pulse_pressure','age_group','hr_reserve','st_angina']

    X = df[feature_cols]
    y = df['target']

    print(f"\n[i] Class distribution: {y.value_counts().to_dict()}")
    print(f"[i] Positive rate: {y.mean():.2%}\n")

    # ── Train/Test Split ──
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42, stratify=y)

    # ── SMOTE Oversampling ──
    smote = SMOTE(random_state=42)
    X_res, y_res = smote.fit_resample(X_train, y_train)
    print(f"[✓] After SMOTE: {dict(zip(*np.unique(y_res, return_counts=True)))}\n")

    # ── Scale ──
    scaler = StandardScaler()
    X_train_sc = scaler.fit_transform(X_res)
    X_test_sc  = scaler.transform(X_test)

    # ── Train All Models ──
    print("[→] Training models with GridSearchCV + 5-fold CV...\n")
    trained_models, cv_results = train_models(X_train_sc, y_res)

    # ── Evaluate ──
    print("\n[→] Evaluating on held-out test set...\n")
    metrics, best_name = evaluate_models(trained_models, X_test_sc, y_test, feature_cols)

    print(f"\n[★] Best model: {best_name}  (AUC={metrics[best_name]['roc_auc']})\n")

    # ── Feature Importance Plot ──
    plot_feature_importance(trained_models[best_name], best_name, feature_cols)

    # ── Summary Report ──
    print("\n── SUMMARY TABLE ──────────────────────────────────────────────")
    print(f"  {'Model':<22} {'Accuracy':>10}  {'ROC-AUC':>10}")
    print("  " + "-"*44)
    for name, m in sorted(metrics.items(), key=lambda x: -x[1]['roc_auc']):
        star = ' ★' if name == best_name else ''
        print(f"  {name+star:<22} {m['accuracy']:>10.4f}  {m['roc_auc']:>10.4f}")
    print()

    # ── Save Models ──
    os.makedirs('outputs', exist_ok=True)
    with open('outputs/models.pkl','wb') as f:
        pickle.dump({'models': trained_models, 'scaler': scaler,
                     'best': best_name, 'features': feature_cols}, f)
    print("[✓] Models saved → outputs/models.pkl")

    # ── Start API Server ──
    print("\n[→] Starting Flask prediction API on http://localhost:5000\n")
    app = create_flask_app(trained_models, best_name, scaler, feature_cols)
    if app:
        app.run(debug=False, port=5000, host='0.0.0.0')
    else:
        print("[!] Flask unavailable — install with: pip install flask flask-cors")


if __name__ == '__main__':
    main()
