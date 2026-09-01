"""
Búsqueda de hiperparámetros para XGBoost con validación temporal.

Usa TimeSeriesSplit sobre el conjunto de TRAIN (nunca toca validación,
para no sobreajustar al examen). El valid final queda intacto como
juez imparcial.
"""
import pandas as pd
import numpy as np
from xgboost import XGBClassifier
from sklearn.model_selection import RandomizedSearchCV, TimeSeriesSplit
from sklearn.metrics import average_precision_score

BASELINE_PR_AUC = 0.2158
XGB_BASE_PR_AUC = 0.3057

df = pd.read_parquet("data/processed/model_data.parquet")

EXCLUDE = ["label", "split", "user_id", "user_session",
           "session_start", "session_end", "first_purchase_ts"]
features = [c for c in df.columns if c not in EXCLUDE]
features = [c for c in features if pd.api.types.is_numeric_dtype(df[c])]

train = df[df.split == "train"].sort_values("session_start")
valid = df[df.split == "valid"]
X_tr, y_tr = train[features], train["label"]
X_va, y_va = valid[features], valid["label"]

# --- Espacio de búsqueda ---
param_dist = {
    "n_estimators": [400, 600, 800, 1000],
    "max_depth": [4, 5, 6, 7, 8],
    "learning_rate": [0.02, 0.03, 0.05, 0.08],
    "subsample": [0.7, 0.8, 0.9],
    "colsample_bytree": [0.7, 0.8, 0.9],
    "min_child_weight": [1, 3, 5, 7],
    "gamma": [0, 0.1, 0.3],
    "reg_lambda": [1, 3, 5],
}

base = XGBClassifier(
    scale_pos_weight=6.59,
    eval_metric="aucpr",
    random_state=42,
    n_jobs=-1,
)

# --- CV temporal: cada fold entrena en pasado, valida en futuro ---
tscv = TimeSeriesSplit(n_splits=4)

search = RandomizedSearchCV(
    base,
    param_distributions=param_dist,
    n_iter=40,                       # 40 combinaciones al azar
    scoring="average_precision",     # PR-AUC
    cv=tscv,
    verbose=2,
    random_state=42,
    n_jobs=-1,
)

print("Buscando... (esto tarda unos minutos)\n")
search.fit(X_tr, y_tr)

print("\n" + "=" * 50)
print("MEJORES HIPERPARÁMETROS")
print("=" * 50)
for k, v in search.best_params_.items():
    print(f"  {k}: {v}")
print(f"\nPR-AUC en CV (train): {search.best_score_:.4f}")

# --- Juez final: el valid intacto ---
best = search.best_estimator_
proba = best.predict_proba(X_va)[:, 1]
pr_auc = average_precision_score(y_va, proba)

print("\n" + "=" * 50)
print("EVALUACIÓN FINAL (valid intacto)")
print("=" * 50)
print(f"Baseline (logística):   {BASELINE_PR_AUC:.4f}")
print(f"XGBoost sin tunear:     {XGB_BASE_PR_AUC:.4f}")
print(f"XGBoost tuneado:        {pr_auc:.4f}")
mejora = 100*(pr_auc-XGB_BASE_PR_AUC)/XGB_BASE_PR_AUC
print(f"\nGanancia del tuning:    {mejora:+.1f}%")

if pr_auc > XGB_BASE_PR_AUC:
    best.save_model("src/models/xgb_model_tuned.json")
    print("\nModelo tuneado guardado (mejoró).")
else:
    print("\nEl tuning NO mejoró. Nos quedamos con el modelo base.")
    print("(Esto también es un resultado válido para la memoria.)")