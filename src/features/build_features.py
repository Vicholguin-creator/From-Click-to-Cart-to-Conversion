"""
Construcción del dataset de modelado.

Población C: sesiones con >=1 cart (intención manifestada).
Filtros de calidad, resolución de colinealidad y split temporal.
"""
import numpy as np
import pandas as pd

RAW = "data/processed/sessions.parquet"
OUT = "data/processed/model_data.parquet"

df = pd.read_parquet(RAW)
print(f"Sesiones totales:        {len(df):,}")

# ---------- 1. Población C ----------
df = df[df.n_carts >= 1].copy()
print(f"Población C:             {len(df):,}")

# ---------- 2. Filtros de calidad ----------
# Sesiones instantáneas: sin comportamiento temporal que modelar
n0 = (df.duration_sec == 0).sum()
df = df[df.duration_sec > 0].copy()
print(f"  - duración 0 seg:      {n0:,} eliminadas")

# Cola larga: sesiones nunca cerradas (hasta 151 días)
p99 = int(df.duration_sec.quantile(0.99))
n_long = (df.duration_sec > p99).sum()
df.loc[df.duration_sec > p99, "duration_sec"] = p99
print(f"  - duración > p99 ({p99/3600:.1f}h): {n_long:,} truncadas")

print(f"Dataset final:           {len(df):,}  |  conversión: {df.label.mean():.2%}\n")

# ---------- 3. Features derivadas (ortogonales) ----------
df["products_per_event"] = df.n_products / df.n_events      # dispersión vs foco
df["brands_per_product"] = df.n_brands / df.n_products      # salto entre marcas
df["log_duration"] = np.log1p(df.duration_sec)              # escala tratable
df["log_events"] = np.log1p(df.n_events)

# ---------- 4. Resolución de colinealidad ----------
DROP = [
    "n_removes",       # rho=0.96 con remove_rate; contaminada por volumen
    "n_products",      # rho=0.92 con n_events; reemplazada por products_per_event
    "n_brands",        # absorbida en brands_per_product
    "cart_rate",       # artefacto: mide brevedad, no intención (rho=-0.07)
    "events_per_min",  # rho=-0.71 con duration_sec; ficticia en sesiones cortas
    "n_views",         # rho=0.67 con n_events
    "price_min",       # redundante con price_avg / price_max
    "duration_sec",    # sustituida por log_duration
]
df = df.drop(columns=DROP)

# ---------- 5. Split TEMPORAL (no aleatorio) ----------
CUT = pd.Timestamp("2020-01-01")
train = df[df.session_start < CUT].copy()
valid = df[df.session_start >= CUT].copy()

print(f"Train (oct-dic 2019): {len(train):>8,}  conversión: {train.label.mean():.2%}")
print(f"Valid (ene-feb 2020): {len(valid):>8,}  conversión: {valid.label.mean():.2%}")

# ---------- 6. Comprobación de deriva ----------
print("\nDeriva temporal (media train vs valid):")
num = train.select_dtypes(include=[np.number]).columns.drop("label")
deriva = pd.DataFrame({
    "train": train[num].mean(),
    "valid": valid[num].mean(),
})
deriva["cambio_%"] = 100 * (deriva.valid - deriva.train) / deriva.train.abs()
print(deriva.sort_values("cambio_%", key=abs, ascending=False).head(8))

# ---------- 7. Guardado ----------
train["split"] = "train"
valid["split"] = "valid"
pd.concat([train, valid]).to_parquet(OUT, index=False)
print(f"\nGuardado en {OUT}")

pos = train.label.sum()
neg = len(train) - pos
print(f"\nscale_pos_weight sugerido para XGBoost: {neg/pos:.2f}")