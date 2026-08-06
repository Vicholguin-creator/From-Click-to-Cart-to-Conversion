"""
Baseline: Regresión Logística.

Punto de comparación honesto antes de XGBoost. Si el modelo complejo
no supera esto de forma clara, no está justificado usarlo.
Métricas para clase desbalanceada: PR-AUC (principal), ROC-AUC, recall.
"""
import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    average_precision_score, roc_auc_score,
    classification_report, confusion_matrix,
)

df = pd.read_parquet("data/processed/model_data.parquet")

# --- Selección EXPLÍCITA de features (nunca drop de label) ---
EXCLUDE = [
    "label", "split",
    "user_id", "user_session",        # identificadores, no predictores
    "session_start", "session_end",   # timestamps → fuga temporal
    "first_purchase_ts",              # define la etiqueta → fuga directa
]
features = [c for c in df.columns if c not in EXCLUDE]
# solo columnas numéricas (por si queda alguna de tipo objeto)
features = [c for c in features if pd.api.types.is_numeric_dtype(df[c])]
print(f"Features usadas ({len(features)}):\n{features}\n")

train = df[df.split == "train"]
valid = df[df.split == "valid"]

X_tr, y_tr = train[features].fillna(0), train["label"]
X_va, y_va = valid[features].fillna(0), valid["label"]

# --- Modelo: escalado + logística con balanceo de clases ---
model = Pipeline([
    ("scaler", StandardScaler()),
    ("clf", LogisticRegression(
        max_iter=1000,
        class_weight="balanced",   # equivalente al scale_pos_weight
        random_state=42,
    )),
])
model.fit(X_tr, y_tr)

# --- Evaluación ---
proba = model.predict_proba(X_va)[:, 1]
pred = (proba >= 0.5).astype(int)

pr_auc = average_precision_score(y_va, proba)
roc_auc = roc_auc_score(y_va, proba)

print("=" * 50)
print("BASELINE — Regresión Logística")
print("=" * 50)
print(f"PR-AUC  (principal): {pr_auc:.4f}")
print(f"ROC-AUC:             {roc_auc:.4f}")
print(f"Tasa base (valid):   {y_va.mean():.4f}")
print(f"\nLift PR-AUC sobre azar: {pr_auc / y_va.mean():.2f}x\n")

print(classification_report(y_va, pred, digits=3))
print("Matriz de confusión (umbral 0.5):")
print(confusion_matrix(y_va, pred))

# --- Coeficientes: qué pesa y en qué dirección ---
coef = pd.Series(
    model.named_steps["clf"].coef_[0], index=features
).sort_values(key=abs, ascending=False)
print("\nCoeficientes (mayor |peso| primero):")
print(coef)