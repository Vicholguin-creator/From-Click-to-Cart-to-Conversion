"""
Modelo 2 — Caracterización de arquetipos (k=4).

Entrena K-Means final, perfila cada cluster y les asigna nombre
de negocio según su comportamiento real. Guarda el modelo y las
asignaciones para uso posterior.
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans

pd.set_option("display.float_format", "{:.3f}".format)
pd.set_option("display.width", 200)

df = pd.read_parquet("data/processed/model_data.parquet")
aband = df[df.label == 0].copy()

FEATURES = [
    "remove_rate", "products_per_event", "n_categories", "n_carts",
    "price_avg", "price_std", "brands_per_product", "log_duration", "log_events",
]
X = aband[FEATURES].fillna(0)

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# --- K-Means final ---
K = 4
km = KMeans(n_clusters=K, random_state=42, n_init=10)
aband["cluster"] = km.fit_predict(X_scaled)

# --- Perfil: media de cada feature por cluster ---
print("=" * 70)
print("PERFIL DE ARQUETIPOS (media por cluster)")
print("=" * 70)
perfil = aband.groupby("cluster")[FEATURES].mean()
perfil["n_sesiones"] = aband.groupby("cluster").size()
perfil["%"] = 100 * perfil["n_sesiones"] / len(aband)
print(perfil.to_string())

# --- Comparación con la media global (para ver qué DESTACA) ---
print("\n" + "=" * 70)
print("DESVIACIÓN vs media global (>1 = por encima; <1 = por debajo)")
print("=" * 70)
global_mean = aband[FEATURES].mean()
desviacion = aband.groupby("cluster")[FEATURES].mean() / global_mean
print(desviacion.to_string())

# --- % de sesiones con remove_rate=0 por cluster (el navegante pasivo) ---
print("\n--- % con remove_rate = 0 (navegante pasivo) por cluster ---")
aband["sin_remove"] = (aband["remove_rate"] == 0).astype(int)
print((100 * aband.groupby("cluster")["sin_remove"].mean()).to_string())

# --- Heatmap visual de los perfiles (z-score) ---
z = (aband.groupby("cluster")[FEATURES].mean() - global_mean) / aband[FEATURES].std()
fig, ax = plt.subplots(figsize=(11, 5))
im = ax.imshow(z.values, cmap="RdBu_r", aspect="auto", vmin=-1.5, vmax=1.5)
ax.set_xticks(range(len(FEATURES)))
ax.set_xticklabels(FEATURES, rotation=45, ha="right")
ax.set_yticks(range(K))
ax.set_yticklabels([f"Cluster {i}\n(n={perfil['n_sesiones'][i]:,})" for i in range(K)])
for i in range(K):
    for j in range(len(FEATURES)):
        ax.text(j, i, f"{z.values[i,j]:.1f}", ha="center", va="center", fontsize=8)
plt.colorbar(im, label="z-score vs media global")
ax.set_title("Perfil de arquetipos de abandono (k=4)")
plt.tight_layout()
plt.savefig("notebooks/kmeans_perfiles.png", dpi=120, bbox_inches="tight")
print("\n→ notebooks/kmeans_perfiles.png")

# --- Guardar asignaciones ---
aband[["user_session", "cluster"]].to_parquet(
    "data/processed/cluster_assignments.parquet", index=False)
print("→ data/processed/cluster_assignments.parquet")