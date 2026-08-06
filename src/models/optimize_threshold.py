import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from xgboost import XGBClassifier
from sklearn.metrics import f1_score, confusion_matrix

df = pd.read_parquet("data/processed/model_data.parquet")
EXCLUDE = ["label", "split", "user_id", "user_session",
           "session_start", "session_end", "first_purchase_ts"]
features = [c for c in df.columns if c not in EXCLUDE]
features = [c for c in features if pd.api.types.is_numeric_dtype(df[c])]

valid = df[df.split == "valid"]
X_va, y_va = valid[features], valid["label"]

model = XGBClassifier()
model.load_model("src/models/xgb_model.json")
proba = model.predict_proba(X_va)[:, 1]

umbrales = np.linspace(0.05, 0.95, 91)

def evaluar(valor_conversion, coste_estimulo):
    filas = []
    for t in umbrales:
        pred = (proba >= t).astype(int)
        tn, fp, fn, tp = confusion_matrix(y_va, pred).ravel()
        coste = fp * coste_estimulo + fn * valor_conversion
        filas.append((t, coste, f1_score(y_va, pred), tp, fp, fn, tn))
    return pd.DataFrame(filas, columns=["umbral", "coste", "f1",
                                        "tp", "fp", "fn", "tn"])

def umbral_cercano(res, valor):
    return res.iloc[(res.umbral - valor).abs().idxmin()]

VALOR_CONVERSION = 30.0
COSTE_ESTIMULO = 2.0

res = evaluar(VALOR_CONVERSION, COSTE_ESTIMULO)
t_coste = res.loc[res.coste.idxmin(), "umbral"]
t_f1 = res.loc[res.f1.idxmax(), "umbral"]

print("=" * 55)
print("UMBRALES OPTIMOS - escenario principal")
print("=" * 55)
print(f"Supuestos: conversion={VALOR_CONVERSION}, estimulo={COSTE_ESTIMULO}")
print(f"Por defecto:              0.50")
print(f"Maximo F1:                {t_f1:.2f}")
print(f"Minimo coste de negocio:  {t_coste:.2f}")

def resumen(t, nombre):
    pred = (proba >= t).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_va, pred).ravel()
    coste = fp * COSTE_ESTIMULO + fn * VALOR_CONVERSION
    print(f"\n--- {nombre} (umbral {t:.2f}) ---")
    print(f"  Intervenciones (tp+fp):     {tp+fp:,}")
    print(f"  Conversiones captadas (tp): {tp:,} de {tp+fn:,}")
    print(f"  Coste total:                {coste:,.0f}")
    print(f"  Recall: {tp/(tp+fn):.3f}   Precision: {tp/(tp+fp):.3f}")

resumen(0.50, "Por defecto")
resumen(t_coste, "Optimo de negocio")

coste_05 = umbral_cercano(res, 0.50)["coste"]
ahorro = coste_05 - res.coste.min()
print(f"\nAhorro vs umbral 0.5: {ahorro:,.0f} sobre {len(y_va):,} sesiones")
print(f"({100*ahorro/coste_05:.1f}% de reduccion de coste)")

print("\n" + "=" * 55)
print("ANALISIS DE SENSIBILIDAD")
print("=" * 55)
print(f"{'valor/coste':>12} {'ratio':>7} {'umbral_opt':>11} {'recall':>8}")

escenarios = [(10, 5), (20, 5), (30, 2), (30, 1), (50, 1), (50, 2), (20, 2)]
sens = []
for v, c in escenarios:
    r = evaluar(v, c)
    t_opt = r.loc[r.coste.idxmin(), "umbral"]
    pred = (proba >= t_opt).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_va, pred).ravel()
    recall = tp / (tp + fn)
    sens.append((v, c, v/c, t_opt, recall))
    print(f"{v:>5}/{c:<5} {v/c:>7.1f} {t_opt:>11.2f} {recall:>8.3f}")

sens_df = pd.DataFrame(sens, columns=["v", "c", "ratio", "umbral", "recall"]).sort_values("ratio")

fig, (ax1, ax3) = plt.subplots(1, 2, figsize=(15, 5))
ax1.plot(res.umbral, res.coste, color="#c44", label="Coste de negocio")
ax1.axvline(t_coste, ls="--", color="#c44", alpha=.6, label=f"Optimo ({t_coste:.2f})")
ax1.axvline(0.5, ls=":", color="gray", label="Umbral 0.5")
ax1.set_xlabel("Umbral"); ax1.set_ylabel("Coste", color="#c44")
ax2 = ax1.twinx()
ax2.plot(res.umbral, res.f1, color="#4a8")
ax2.set_ylabel("F1", color="#4a8")
ax1.set_title("Coste de negocio y F1 segun umbral")
ax1.legend(loc="upper right")
ax3.plot(sens_df.ratio, sens_df.umbral, "o-", color="#48c")
ax3.set_xlabel("Ratio valor / coste")
ax3.set_ylabel("Umbral optimo")
ax3.set_title("Sensibilidad del umbral")
plt.tight_layout()
plt.savefig("notebooks/umbral_optimo.png", dpi=120)
print("\nGuardado: notebooks/umbral_optimo.png")
