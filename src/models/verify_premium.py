"""
Verificación del Cluster 3 (premium): ¿los precios altos son reales
o un artefacto de datos (outliers, precios mal cargados)?
"""
import pandas as pd
import numpy as np

df = pd.read_parquet("data/processed/model_data.parquet")
aband = df[df.label == 0].copy()

# Reconstruir la asignación de clusters guardada
clusters = pd.read_parquet("data/processed/cluster_assignments.parquet")
aband = aband.merge(clusters, on="user_session", how="left")

# --- 1. Distribución de price_avg global vs Cluster 3 ---
print("=" * 60)
print("price_avg — GLOBAL (todas las sesiones de abandono)")
print("=" * 60)
print(aband["price_avg"].describe(percentiles=[.5, .9, .95, .99, .999]))

c3 = aband[aband.cluster == 3]
print("\n" + "=" * 60)
print(f"price_avg — CLUSTER 3 (n={len(c3):,})")
print("=" * 60)
print(c3["price_avg"].describe(percentiles=[.5, .9, .95, .99, .999]))

# --- 2. ¿Son precios plausibles para cosmética? ---
print("\n--- Rangos de price_avg en Cluster 3 ---")
tramos = pd.cut(c3["price_avg"],
                [0, 50, 100, 200, 500, 1000, np.inf])
print(c3.groupby(tramos, observed=True).size())

# --- 3. price_max: ¿hay valores absurdos? ---
print("\n--- price_max en Cluster 3 ---")
print(c3["price_max"].describe(percentiles=[.5, .9, .99, .999]))

# --- 4. ¿Cuántas sesiones tienen precios 'imposibles'? ---
absurdo = (c3["price_avg"] > 1000).sum()
print(f"\nSesiones con price_avg > 1000: {absurdo:,} "
      f"({100*absurdo/len(c3):.2f}% del cluster)")

# --- 5. Comparar con el precio global del dataset crudo ---
print("\n--- Contexto: price_avg de TODA la población (compran y no) ---")
print(df["price_avg"].describe(percentiles=[.5, .95, .99]))