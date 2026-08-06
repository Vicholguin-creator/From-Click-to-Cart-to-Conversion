"""
Interpretabilidad con SHAP sobre el modelo XGBoost final.

Objetivo central: entender la FORMA de la relación de cada feature,
en especial remove_rate (rank 1 en XGBoost, ~0 en la logística).
SHAP explica por qué el modelo de árboles capturó lo que el lineal perdió.
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import shap
from xgboost import XGBClassifier

df = pd.read_parquet("data/processed/model_data.parquet")

EXCLUDE = ["label", "split", "user_id", "user_session",
           "session_start", "session_end", "first_purchase_ts"]
features = [c for c in df.columns if c not in EXCLUDE]
features = [c for c in features if pd.api.types.is_numeric_dtype(df[c])]

valid = df[df.split == "valid"]
X_va = valid[features]

# Cargar el modelo final
model = XGBClassifier()
model.load_model("src/models/xgb_model.json")

# --- Calcular valores SHAP (muestra para velocidad) ---
sample = X_va.sample(n=min(20000, len(X_va)), random_state=42)
explainer = shap.TreeExplainer(model)
shap_values = explainer.shap_values(sample)
print(f"SHAP calculado sobre {len(sample):,} sesiones de validación\n")

# --- 1. Resumen: importancia + dirección ---
plt.figure()
shap.summary_plot(shap_values, sample, show=False, max_display=15)
plt.tight_layout()
plt.savefig("notebooks/shap_summary.png", dpi=120, bbox_inches="tight")
plt.close()
print("→ notebooks/shap_summary.png")

# --- 2. Barra: importancia media absoluta ---
plt.figure()
shap.summary_plot(shap_values, sample, plot_type="bar",
                  show=False, max_display=15)
plt.tight_layout()
plt.savefig("notebooks/shap_bar.png", dpi=120, bbox_inches="tight")
plt.close()
print("→ notebooks/shap_bar.png")

# --- 3. Dependencia de remove_rate: LA FORMA de la relación ---
plt.figure()
shap.dependence_plot("remove_rate", shap_values, sample,
                     show=False, interaction_index=None)
plt.tight_layout()
plt.savefig("notebooks/shap_remove_rate.png", dpi=120, bbox_inches="tight")
plt.close()
print("→ notebooks/shap_remove_rate.png")

# --- 4. Dependencia de las otras dos top features ---
for feat in ["products_per_event", "n_events"]:
    plt.figure()
    shap.dependence_plot(feat, shap_values, sample,
                         show=False, interaction_index=None)
    plt.tight_layout()
    plt.savefig(f"notebooks/shap_{feat}.png", dpi=120, bbox_inches="tight")
    plt.close()
    print(f"→ notebooks/shap_{feat}.png")

# --- 5. Importancia media |SHAP| en números ---
mean_abs = pd.Series(
    np.abs(shap_values).mean(axis=0), index=features
).sort_values(ascending=False)
print("\nImportancia media |SHAP| (impacto en la predicción):")
print(mean_abs)

# --- 6. remove_rate: dirección del efecto por tramos ---
print("\n--- Efecto de remove_rate por tramos ---")
tmp = pd.DataFrame({
    "remove_rate": sample["remove_rate"].values,
    "shap": shap_values[:, features.index("remove_rate")],
})
tmp["tramo"] = pd.cut(tmp.remove_rate,
                      [-0.01, 0, 0.25, 0.5, 1.0, np.inf],
                      labels=["=0", "0-0.25", "0.25-0.5", "0.5-1.0", ">1.0"])
print(tmp.groupby("tramo", observed=True).agg(
    n=("shap", "size"),
    shap_medio=("shap", "mean"),
))
print("\nSHAP positivo = empuja hacia compra; negativo = hacia abandono")