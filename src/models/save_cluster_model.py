"""
el modelo K-Means (Modelo 2) y su scaler para la app.
Reentrena con los mismos parámetros que profile_clusters.py y persiste
los objetos con joblib, además de un mapa cluster->nombre de arquetipo.
"""
import pandas as pd
import numpy as np
import joblib
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans

df = pd.read_parquet("data/processed/model_data.parquet")
aband = df[df.label == 0].copy()

FEATURES = [
    "remove_rate", "products_per_event", "n_categories", "n_carts",
    "price_avg", "price_std", "brands_per_product", "log_duration", "log_events",
]
X = aband[FEATURES].fillna(0)

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

km = KMeans(n_clusters=4, random_state=42, n_init=10)
km.fit(X_scaled)

# Mapa cluster -> nombre de arquetipo (según el análisis que hicimos)
# OJO: el número de cluster puede variar entre ejecuciones. Lo asignamos
# por su perfil real, no por su índice.
aband["cluster"] = km.labels_
perfil = aband.groupby("cluster")[FEATURES].mean()

nombres = {}
for c in perfil.index:
    row = perfil.loc[c]
    # Firma de cada arquetipo según los rasgos dominantes
    if row["price_avg"] > 40:                      # precios muy altos
        nombres[c] = "Comprador premium"
    elif row["brands_per_product"] > 0.7:          # salta poco entre marcas
        nombres[c] = "Monomarca"
    elif row["n_categories"] > 5:                  # explora mucho
        nombres[c] = "Explorador intensivo"
    else:
        nombres[c] = "Navegante pasivo"

print("Mapa de arquetipos:")
for c, n in nombres.items():
    print(f"  Cluster {c}: {n}  (price_avg={perfil.loc[c,'price_avg']:.1f}, "
          f"brands={perfil.loc[c,'brands_per_product']:.2f}, "
          f"cats={perfil.loc[c,'n_categories']:.1f})")

# Guardar todo lo que la app necesita
joblib.dump(scaler, "src/models/cluster_scaler.joblib")
joblib.dump(km, "src/models/cluster_kmeans.joblib")
joblib.dump({"features": FEATURES, "nombres": nombres},
            "src/models/cluster_meta.joblib")

print("\nGuardados:")
print("  src/models/cluster_scaler.joblib")
print("  src/models/cluster_kmeans.joblib")
print("  src/models/cluster_meta.joblib")