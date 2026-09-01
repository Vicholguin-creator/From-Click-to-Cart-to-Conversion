"""
Modelo 3 — Clasificador de probabilidad de retorno (RFM).

Entrena DOS versiones para separar el poder de RFM puro del de
"ya era comprador":
  A) RFM completo (con total_purchases)
  B) RFM sin historial de compra (recency, frequency, actividad)
Produce un score de probabilidad → narrativa de 3 grupos de negocio.
"""
import pandas as pd
import numpy as np
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    average_precision_score, roc_auc_score,
    classification_report, confusion_matrix,
)

rfm = pd.read_parquet("data/processed/rfm_users.parquet")
print(f"Usuarios: {len(rfm):,} | tasa de retorno: {rfm.volvio.mean():.2%}\n")

# Desbalance para XGBoost
neg, pos = (rfm.volvio == 0).sum(), (rfm.volvio == 1).sum()
spw = neg / pos
print(f"scale_pos_weight: {spw:.2f}\n")

y = rfm["volvio"]

# --- Dos conjuntos de features ---
FEATS_A = ["recency_days", "frequency", "monetary",
           "total_carts", "total_purchases", "avg_events"]      # completo
FEATS_B = ["recency_days", "frequency", "avg_events"]           # RFM puro sin compra

def entrenar(feats, nombre):
    X = rfm[feats].fillna(0)
    # split aleatorio estratificado (aquí NO hay orden temporal:
    # RFM ya se calculó en observación, etiqueta en futuro)
    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=0.25, stratify=y, random_state=42
    )
    model = XGBClassifier(
        n_estimators=300, max_depth=5, learning_rate=0.05,
        subsample=0.8, colsample_bytree=0.8,
        scale_pos_weight=spw, eval_metric="aucpr",
        random_state=42, n_jobs=-1,
    )
    model.fit(X_tr, y_tr)
    proba = model.predict_proba(X_te)[:, 1]
    pred = (proba >= 0.5).astype(int)

    pr = average_precision_score(y_te, proba)
    roc = roc_auc_score(y_te, proba)
    print("=" * 55)
    print(f"VERSIÓN {nombre}  ({len(feats)} features)")
    print("=" * 55)
    print(f"PR-AUC:  {pr:.4f}   ROC-AUC: {roc:.4f}   base: {y_te.mean():.4f}")
    print(f"Lift PR-AUC sobre azar: {pr/y_te.mean():.2f}x")
    imp = pd.Series(model.feature_importances_, index=feats).sort_values(ascending=False)
    print("\nImportancia:")
    print(imp.to_string())
    print()
    return model, pr, (X_te, y_te, proba)

modelA, prA, dataA = entrenar(FEATS_A, "A — RFM completo")
modelB, prB, dataB = entrenar(FEATS_B, "B — RFM puro (sin compra)")

print("=" * 55)
print("COMPARACIÓN")
print("=" * 55)
print(f"A (con total_purchases): PR-AUC {prA:.4f}")
print(f"B (RFM puro):            PR-AUC {prB:.4f}")
print(f"Aporte del historial de compra: {100*(prA-prB)/prB:+.1f}%")
print("\n→ Si B ya es fuerte, RFM predice el retorno por sí mismo,")
print("  no solo por 'ya era comprador'.\n")

# --- Score y 3 grupos de negocio (usando el modelo completo) ---
X_te, y_te, proba = dataA
res = pd.DataFrame({"volvio_real": y_te.values, "score": proba})

# Terciles del score → 3 grupos
res["grupo"] = pd.qcut(res.score, 3, labels=["Baja", "Media", "Alta"])
print("=" * 55)
print("3 GRUPOS DE NEGOCIO (por score de retorno)")
print("=" * 55)
g = res.groupby("grupo", observed=True).agg(
    usuarios=("score", "size"),
    score_medio=("score", "mean"),
    retorno_real=("volvio_real", "mean"),
)
print(g.to_string())
print("\nAlta  = vuelve solo → no invertir")
print("Media = franja de intervención → aquí rinde el estímulo")
print("Baja  = perdido → no malgastar")

modelA.save_model("src/models/recovery_model.json")
print("\nModelo guardado en src/models/recovery_model.json")