"""
Modelo 3 — Variables RFM a nivel de usuario + etiqueta de retorno.

División temporal para evitar fuga:
  - Observación (oct-dic 2019): se calcula RFM
  - Futuro (ene-feb 2020): ¿el usuario volvió a tener actividad?
Solo usuarios activos en observación. Etiqueta: reapareció en futuro.
"""
import pandas as pd
import numpy as np

# sessions.parquet tiene TODAS las sesiones (antes del filtro población C)
s = pd.read_parquet("data/processed/sessions.parquet")
print(f"Sesiones totales: {len(s):,}")

CUT = pd.Timestamp("2020-01-01")
obs = s[s.session_start < CUT].copy()      # observación
fut = s[s.session_start >= CUT].copy()     # futuro

print(f"Observación (oct-dic): {len(obs):,} sesiones")
print(f"Futuro (ene-feb):      {len(fut):,} sesiones\n")

# --- Referencia para Recency: fin del periodo de observación ---
ref = obs.session_start.max()

# --- Variables RFM por usuario (solo con datos de observación) ---
rfm = obs.groupby("user_id").agg(
    # Recency: días desde su última sesión hasta el corte (menor = más reciente)
    last_seen=("session_start", "max"),
    # Frequency: nº de sesiones en observación
    frequency=("user_session", "count"),
    # Monetary (aproximada): valor medio del carrito explorado
    monetary=("price_avg", "mean"),
    # Señales de comportamiento útiles
    total_carts=("n_carts", "sum"),
    total_purchases=("label", "sum"),
    avg_events=("n_events", "mean"),
).reset_index()

rfm["recency_days"] = (ref - rfm.last_seen).dt.total_seconds() / 86400
rfm = rfm.drop(columns="last_seen")

# --- Etiqueta: ¿volvió en el periodo futuro? ---
usuarios_futuro = set(fut.user_id.unique())
rfm["volvio"] = rfm.user_id.isin(usuarios_futuro).astype(int)

print(f"Usuarios en observación: {len(rfm):,}")
print(f"Volvieron (ene-feb):     {rfm.volvio.sum():,} ({rfm.volvio.mean():.2%})\n")

# --- Perfil RFM: cómo se ven los que vuelven vs los que no ---
print("=" * 60)
print("PERFIL: media por clase")
print("=" * 60)
cols = ["recency_days", "frequency", "monetary", "total_carts",
        "total_purchases", "avg_events"]
print(rfm.groupby("volvio")[cols].mean().to_string())

# --- Guardar ---
rfm.to_parquet("data/processed/rfm_users.parquet", index=False)
print(f"\nGuardado en data/processed/rfm_users.parquet")
print(f"Columnas: {list(rfm.columns)}")