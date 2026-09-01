"""
Modelo principal: XGBoost.

Mismo split temporal y mismas features que la baseline, para
comparación justa. scale_pos_weight=6.59 para el desbalance.
Objetivo: superar el PR-AUC de la baseline (0.2158).
"""
import pandas as pd
import numpy as np
from xgboost import XGBClassifier
from sklearn.metrics import (
    average_precision_score, roc_auc_score,
    classification_report, confusion_matrix,
    precision_recall_curve,
)

BASELINE_PR_AUC = 0.2158

df = pd.read_parquet("data/processed/model_data.parquet")

EXCLUDE = [
    "label", "split",
    "user_id", "user_session",
    "session_start", "session_end",
    "first_purchase_ts",
]
features = [c for c in df.columns if c not in EXCLUDE]
features = [c for c in features if pd.api.types.is_numeric_dtype(df[c])]

train = df[df.split == "train"]
valid = df[df.split == "valid"]
X_tr, y_tr = train[features], train["label"]
X_va, y_va = valid[features], valid["label"]

# --- XGBoost ---
model = XGBClassifier(
    n_estimators=400,
    max_depth=5,
    learning_rate=0.05,
    subsample=0.8,
    colsample_bytree=0.8,
    scale_pos_weight=6.59,       # desbalance de la clase positiva
    eval_metric="aucpr",         # optimiza PR-AUC directamente
    early_stopping_rounds=30,
    random_state=42,
    n_jobs=-1,
)
model.fit(X_tr, y_tr, eval_set=[(X_va, y_va)], verbose=False)
print(f"Mejor iteración: {model.best_iteration}")

# --- Evaluación ---
proba = model.predict_proba(X_va)[:, 1]
pred = (proba >= 0.5).astype(int)

pr_auc = average_precision_score(y_va, proba)
roc_auc = roc_auc_score(y_va, proba)

print("=" * 50)
print("XGBoost")
print("=" * 50)
print(f"PR-AUC  (principal): {pr_auc:.4f}")
print(f"ROC-AUC:             {roc_auc:.4f}")
print(f"Tasa base (valid):   {y_va.mean():.4f}\n")

print(f"Baseline PR-AUC:     {BASELINE_PR_AUC:.4f}")
print(f"Mejora sobre baseline: {100*(pr_auc-BASELINE_PR_AUC)/BASELINE_PR_AUC:+.1f}%")
print(f"Lift sobre azar:       {pr_auc / y_va.mean():.2f}x\n")

print(classification_report(y_va, pred, digits=3))
print("Matriz de confusión (umbral 0.5):")
print(confusion_matrix(y_va, pred))

# --- Importancia de features (ganancia) ---
imp = pd.Series(
    model.feature_importances_, index=features
).sort_values(ascending=False)
print("\nImportancia de features (gain):")
print(imp)

# --- ¿Recupera remove_rate? ---
print(f"\nRanking de remove_rate: "
      f"{list(imp.index).index('remove_rate')+1} de {len(features)}")

# --- Guardar modelo ---
model.save_model("src/models/xgb_model.json")
print("\nModelo guardado en src/models/xgb_model.json")