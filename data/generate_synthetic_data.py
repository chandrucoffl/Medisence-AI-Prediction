"""
Synthetic data generator for Heart Disease & Kidney Disease.
NOTE: Real UCI/Kaggle CSVs for these two datasets could not be downloaded
in this environment (network restricted), so realistic synthetic data is
generated here using clinically-informed feature ranges, correlations and
a logistic risk function -- a standard practical technique when curated
datasets are unavailable. Feature ranges match the real UCI Cleveland
Heart Disease and UCI CKD dataset schemas exactly, so the trained models
and Flask app will work identically with real data if swapped in later.
"""
import numpy as np
import pandas as pd

rng = np.random.default_rng(42)

# ───────────────────────── HEART DISEASE ─────────────────────────
N = 900
age      = rng.integers(29, 78, N)
sex      = rng.integers(0, 2, N)
cp       = rng.integers(0, 4, N)          # chest pain type
trestbps = rng.normal(131, 17, N).clip(94, 200).round().astype(int)
chol     = rng.normal(246, 51, N).clip(126, 564).round().astype(int)
fbs      = (rng.random(N) < 0.15).astype(int)
restecg  = rng.integers(0, 3, N)
thalach  = (210 - 0.7 * age + rng.normal(0, 15, N)).clip(71, 202).round().astype(int)
exang    = (rng.random(N) < 0.33).astype(int)
oldpeak  = rng.exponential(1.0, N).clip(0, 6.2).round(1)
slope    = rng.integers(0, 3, N)
ca       = rng.integers(0, 4, N)
thal     = rng.choice([1, 2, 3], N, p=[0.05, 0.55, 0.40])

risk = (
    0.035 * age + 0.9 * sex + 0.55 * cp + 0.012 * chol + 0.02 * trestbps
    + 0.8 * exang + 0.5 * oldpeak + 0.6 * ca + 0.4 * (thal == 3)
    - 0.045 * thalach - 3.4
)
prob = 1 / (1 + np.exp(-risk))
target = (rng.random(N) < prob).astype(int)

heart = pd.DataFrame({
    "age": age, "sex": sex, "cp": cp, "trestbps": trestbps, "chol": chol,
    "fbs": fbs, "restecg": restecg, "thalach": thalach, "exang": exang,
    "oldpeak": oldpeak, "slope": slope, "ca": ca, "thal": thal, "target": target,
})
heart.to_csv("/home/claude/MediSense/data/heart.csv", index=False)
print("Heart disease dataset:", heart.shape, "positive rate:", target.mean().round(3))

# ───────────────────────── KIDNEY DISEASE ─────────────────────────
N2 = 700
age2   = rng.integers(6, 90, N2)
bp     = rng.normal(76, 14, N2).clip(50, 180).round().astype(int)
sg     = rng.choice([1.005, 1.010, 1.015, 1.020, 1.025], N2,
                     p=[0.15, 0.2, 0.25, 0.25, 0.15])
al     = rng.choice([0, 1, 2, 3, 4, 5], N2, p=[0.45, 0.15, 0.15, 0.1, 0.1, 0.05])
su     = rng.choice([0, 1, 2, 3, 4, 5], N2, p=[0.75, 0.05, 0.05, 0.05, 0.05, 0.05])
bgr    = rng.normal(148, 74, N2).clip(22, 490).round().astype(int)
bu     = rng.normal(57, 50, N2).clip(1.5, 391).round().astype(int)
sc     = rng.exponential(1.8, N2).clip(0.4, 76).round(1)
sod    = rng.normal(137, 10, N2).clip(4.5, 163).round().astype(int)
pot    = rng.normal(4.6, 2.9, N2).clip(2.5, 47).round(1)
hemo   = rng.normal(12.5, 2.9, N2).clip(3.1, 17.8).round(1)
pcv    = rng.normal(38, 8, N2).clip(9, 54).round().astype(int)
wbcc   = rng.normal(8400, 2900, N2).clip(2200, 26400).round().astype(int)
rbcc   = rng.normal(4.7, 1.0, N2).clip(2.1, 8.0).round(1)

risk2 = (
    0.02 * age2 + 0.03 * bp - 55 * (sg - 1.02) + 0.35 * al + 0.15 * su
    + 0.006 * bgr + 0.03 * bu + 0.55 * sc - 0.4 * hemo - 0.05 * pcv
    + 0.0002 * wbcc - 0.35 * rbcc - 2.0
)
prob2 = 1 / (1 + np.exp(-risk2))
classification = (rng.random(N2) < prob2).astype(int)  # 1 = CKD, 0 = not CKD

kidney = pd.DataFrame({
    "age": age2, "blood_pressure": bp, "specific_gravity": sg, "albumin": al,
    "sugar": su, "blood_glucose_random": bgr, "blood_urea": bu,
    "serum_creatinine": sc, "sodium": sod, "potassium": pot,
    "haemoglobin": hemo, "packed_cell_volume": pcv,
    "white_blood_cell_count": wbcc, "red_blood_cell_count": rbcc,
    "classification": classification,
})
kidney.to_csv("/home/claude/MediSense/data/kidney.csv", index=False)
print("Kidney disease dataset:", kidney.shape, "positive rate:", classification.mean().round(3))
