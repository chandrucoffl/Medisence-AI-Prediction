"""
Train one production-quality model per disease and package everything
(model + scaler + feature schema + metadata) into model/diseases.pkl,
which app.py loads at startup.
"""
import pickle, os
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.datasets import load_breast_cancer

BASE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(BASE, "..", "data")

DISEASES = {}

def fit_and_package(key, X, y, feature_names, model, name, icon, color, description):
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y)
    scaler = StandardScaler()
    X_train_sc = scaler.fit_transform(X_train)
    X_test_sc  = scaler.transform(X_test)
    model.fit(X_train_sc, y_train)
    preds = model.predict(X_test_sc)
    proba = model.predict_proba(X_test_sc)[:, 1]
    acc = accuracy_score(y_test, preds)
    auc = roc_auc_score(y_test, proba)
    DISEASES[key] = {
        "model": model, "scaler": scaler, "feature_names": list(feature_names),
        "name": name, "icon": icon, "color": color, "description": description,
        "accuracy": round(acc * 100, 2), "auc": round(auc * 100, 2),
        "positive_label": "High Risk", "negative_label": "Low Risk",
    }
    print(f"  {icon} {name:<22} Accuracy: {acc*100:5.2f}%   AUC: {auc*100:5.2f}%")


print("Training all disease models...\n")

# 1. Breast Cancer (sklearn built-in, real data)
bc = load_breast_cancer()
fit_and_package(
    "breast_cancer", bc.data, bc.target, bc.feature_names,
    RandomForestClassifier(n_estimators=200, random_state=42),
    "Breast Cancer Risk", "🎗️", "#dc3545",
    "Predicts malignant vs benign tumor risk from cell nuclei measurements (real UCI/sklearn data).",
)
# NOTE: sklearn breast cancer target: 0 = malignant, 1 = benign -> invert for "High Risk" semantics
DISEASES["breast_cancer"]["invert_positive"] = True

# 2. Diabetes (real Pima Indians UCI data)
diab = pd.read_csv(os.path.join(DATA, "diabetes.csv"))
X = diab.drop(columns=["Outcome"]).values
y = diab["Outcome"].values
fit_and_package(
    "diabetes", X, y, diab.drop(columns=["Outcome"]).columns,
    GradientBoostingClassifier(n_estimators=200, max_depth=3, random_state=42),
    "Diabetes Risk", "🩸", "#fd7e14",
    "Predicts Type-2 diabetes risk from glucose, BMI, insulin and related clinical markers (real UCI data).",
)

# 3. Heart Disease (synthetic, UCI-schema-matched)
heart = pd.read_csv(os.path.join(DATA, "heart.csv"))
X = heart.drop(columns=["target"]).values
y = heart["target"].values
fit_and_package(
    "heart_disease", X, y, heart.drop(columns=["target"]).columns,
    RandomForestClassifier(n_estimators=250, max_depth=8, random_state=42),
    "Heart Disease Risk", "❤️", "#e83e8c",
    "Predicts coronary heart disease risk from ECG, cholesterol and exercise-test features.",
)

# 4. Kidney Disease (synthetic, UCI-CKD-schema-matched)
kidney = pd.read_csv(os.path.join(DATA, "kidney.csv"))
X = kidney.drop(columns=["classification"]).values
y = kidney["classification"].values
fit_and_package(
    "kidney_disease", X, y, kidney.drop(columns=["classification"]).columns,
    LogisticRegression(max_iter=5000, random_state=42),
    "Kidney Disease Risk", "🫘", "#20c997",
    "Predicts chronic kidney disease (CKD) risk from blood, urine and electrolyte panel values.",
)

with open(os.path.join(BASE, "diseases.pkl"), "wb") as f:
    pickle.dump(DISEASES, f)

print("\n✅ All 4 disease models saved to model/diseases.pkl")
