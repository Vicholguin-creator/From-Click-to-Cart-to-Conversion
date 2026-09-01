"""
Modelo 2 — Arquetipos de abandono (K-Means).

Clusteriza las sesiones de Población C SIN compra (label=0):
mostraron intención (añadieron al carrito) pero no convirtieron.
Objetivo: identificar tipos de abandono diferenciables y accionables.
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

df = pd.read_parquet("data/processed/model_data.parquet")

# --- Solo abandonos: intención manifestada, sin compra ---
aband = df[df.label == 0].copy()
print(f"Sesiones de abandono (Población C, label=0): {len(aband):,}\n")

# --- Features de COMPORTAMIENTO (no de volumen bruto) ---
# Describen CÓMO se comportó la sesión, para separar arquetipos
FEATURES = [
    "remove_rate",          # ¿editó el carrito? (0 = navegante pasivo)
    "products_per_event",   # foco vs dispersión
    "n_categories",         # amplitud de exploración
    "n_carts",              # intensidad de intención
    "price_avg",            # nivel de precio explorado
    "price_std",            # ¿compara precios distintos?
    "brands_per_product",   # ¿salta entre marcas?
    "log_duration",         # tiempo invertido
    "log_events",           # intensidad de interacción
]
X = aband[FEATURES].fillna(0)

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# --- Selección de k: codo + silueta ---
# Silueta sobre muestra (es cara de calcular en 800K filas)
rng = np.random.RandomState(42)
sample_idx = rng.choice(len(X_scaled), size=min(30000, len(X_scaled)), replace=False)
X_sample = X_scaled[sample_idx]

print("Evaluando k de 2 a 8...\n")
print(f"{'k':>3} {'inercia':>14} {'silueta':>10}")
inercias, siluetas = [], []
for k in range(2, 9):
    km = KMeans(n_clusters=k, random_state=42, n_init=10)
    labels_full = km.fit_predict(X_scaled)
    sil = silhouette_score(X_sample, labels_full[sample_idx])
    inercias.append(km.inertia_)
    siluetas.append(sil)
    print(f"{k:>3} {km.inertia_:>14,.0f} {sil:>10.4f}")

# --- Gráfico codo + silueta ---
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))
ax1.plot(range(2, 9), inercias, "o-", color="#4a8")
ax1.set_xlabel("k"); ax1.set_ylabel("Inercia")
ax1.set_title("Método del codo")
ax2.plot(range(2, 9), siluetas, "o-", color="#c44")
ax2.set_xlabel("k"); ax2.set_ylabel("Silueta")
ax2.set_title("Coeficiente de silueta")
plt.tight_layout()
plt.savefig("notebooks/kmeans_seleccion_k.png", dpi=120)
print("\n→ notebooks/kmeans_seleccion_k.png")

print("\nElige k mirando: dónde se 'dobla' la inercia y dónde la silueta")
print("es más alta. Un k entre 3 y 5 suele ser lo interpretable.")