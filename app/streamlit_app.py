"""
App de productivizacion - Sistema de intencion y recuperacion retail.
Dos modos: exploracion individual (sliders) y prediccion por lotes.
La prediccion por lotes acepta datos propios via carga de CSV.
"""
import streamlit as st
import numpy as np
import pandas as pd
import joblib
from xgboost import XGBClassifier

st.set_page_config(page_title="Intencion de compra retail", layout="wide")

UMBRAL = 0.33
XGB_FEATURES = ["n_events", "n_carts", "n_categories", "price_avg", "price_max",
                "price_std", "brand_null_ratio", "hour", "dayofweek", "is_weekend",
                "remove_rate", "products_per_event", "brands_per_product",
                "log_duration", "log_events"]
ACCIONES = {
    "Navegante pasivo": "Sin intencion real. No invertir en retencion.",
    "Explorador intensivo": "Abandono activo. Recordatorio de carrito o incentivo.",
    "Monomarca": "Cliente fiel a una marca. Fidelizacion, no persuasion de precio.",
    "Comprador premium": "Alto valor. Intervencion de precio (descuento, envio gratis).",
}

@st.cache_resource
def cargar_modelos():
    xgb = XGBClassifier()
    xgb.load_model("src/models/xgb_model.json")
    scaler = joblib.load("src/models/cluster_scaler.joblib")
    kmeans = joblib.load("src/models/cluster_kmeans.joblib")
    meta = joblib.load("src/models/cluster_meta.joblib")
    return xgb, scaler, kmeans, meta

xgb, scaler, kmeans, meta = cargar_modelos()
CLUSTER_FEATURES = meta["features"]
NOMBRES = meta["nombres"]

@st.cache_data
def cargar_validacion():
    df = pd.read_parquet("data/processed/model_data.parquet")
    return df[df.split == "valid"].copy()

def asignar_arquetipo(df):
    x = df[CLUSTER_FEATURES].fillna(0)
    clusters = kmeans.predict(scaler.transform(x))
    return pd.Series(clusters, index=df.index).map(NOMBRES)

def procesar_lote(muestra):
    """Aplica ambos modelos a un DataFrame y devuelve resultados."""
    muestra = muestra.copy()
    proba_lote = xgb.predict_proba(muestra[XGB_FEATURES].fillna(0))[:, 1]
    muestra["prob_compra"] = proba_lote
    muestra["intervenir"] = np.where(proba_lote >= UMBRAL, "SI", "no")
    muestra["arquetipo"] = asignar_arquetipo(muestra)
    muestra["accion"] = muestra["arquetipo"].map(ACCIONES)
    return muestra

def mostrar_resultados(muestra):
    c1, c2, c3 = st.columns(3)
    c1.metric("Sesiones procesadas", f"{len(muestra):,}")
    c2.metric("Alta intencion", f"{(muestra.intervenir=='SI').sum():,}")
    c3.metric("% a intervenir", f"{100*(muestra.intervenir=='SI').mean():.1f}%")
    st.write("**Distribucion de arquetipos:**")
    st.bar_chart(muestra["arquetipo"].value_counts())
    st.write("**Lista priorizada de intervencion:**")
    cols_tabla = ["prob_compra", "intervenir", "arquetipo", "accion",
                  "n_events", "n_carts", "remove_rate", "price_avg"]
    tabla = muestra.sort_values("prob_compra", ascending=False)[cols_tabla].head(50)
    tabla["prob_compra"] = (tabla["prob_compra"] * 100).round(1)
    st.dataframe(tabla, width="stretch")
    csv = muestra[["prob_compra", "intervenir", "arquetipo", "accion"]].to_csv(index=False)
    st.download_button("Descargar resultados (CSV)", csv,
                       "predicciones_lote.csv", "text/csv")

st.title("Prediccion de intencion de compra y arquetipo de abandono")
st.caption("TFM - From Click to Cart to Conversion - Dataset REES46")

tab1, tab2 = st.tabs(["Exploracion individual", "Prediccion por lotes (negocio)"])

# ============================================================
# TAB 1 - SLIDERS
# ============================================================
with tab1:
    col_in, col_out = st.columns([1, 1])
    with col_in:
        st.subheader("Datos de la sesion")
        n_events = st.slider("N de eventos", 1, 100, 8)
        n_carts = st.slider("N de veces que anadio al carrito", 1, 20, 3)
        n_removes = st.slider("N de veces que quito del carrito", 0, 20, 0)
        n_products = st.slider("N de productos distintos vistos", 1, 50, 5)
        n_categories = st.slider("N de categorias exploradas", 1, 15, 3)
        n_brands = st.slider("N de marcas distintas", 1, 15, 2)
        price_avg = st.slider("Precio medio explorado (EUR)", 1.0, 200.0, 15.0)
        price_std = st.slider("Dispersion de precio (EUR)", 0.0, 100.0, 5.0)
        duration_min = st.slider("Duracion de la sesion (min)", 0.5, 120.0, 6.0)
        hour = st.slider("Hora del dia", 0, 23, 14)

    remove_rate = n_removes / n_carts if n_carts > 0 else 0.0
    products_per_event = n_products / n_events
    brands_per_product = n_brands / n_products
    duration_sec = duration_min * 60
    log_duration = np.log1p(duration_sec)
    log_events = np.log1p(n_events)
    price_max = price_avg + price_std

    x1 = pd.DataFrame([{
        "n_events": n_events, "n_carts": n_carts, "n_categories": n_categories,
        "price_avg": price_avg, "price_max": price_max, "price_std": price_std,
        "brand_null_ratio": 0.0, "hour": hour, "dayofweek": 3, "is_weekend": 0,
        "remove_rate": remove_rate, "products_per_event": products_per_event,
        "brands_per_product": brands_per_product,
        "log_duration": log_duration, "log_events": log_events,
    }])[XGB_FEATURES]
    proba = float(xgb.predict_proba(x1)[:, 1][0])

    x2 = pd.DataFrame([{
        "remove_rate": remove_rate, "products_per_event": products_per_event,
        "n_categories": n_categories, "n_carts": n_carts, "price_avg": price_avg,
        "price_std": price_std, "brands_per_product": brands_per_product,
        "log_duration": log_duration, "log_events": log_events,
    }])[CLUSTER_FEATURES]
    cluster = int(kmeans.predict(scaler.transform(x2))[0])
    arquetipo = NOMBRES[cluster]

    with col_out:
        st.subheader("Prediccion")
        st.metric("Probabilidad de compra", f"{proba:.1%}")
        if proba >= UMBRAL:
            st.success(f"Alta intencion (>= {UMBRAL:.0%}). Vale la pena intervenir.")
        else:
            st.warning(f"Baja intencion (< {UMBRAL:.0%}).")
        st.divider()
        st.subheader("Arquetipo de abandono")
        st.info(f"**{arquetipo}**")
        st.write("**Accion recomendada:**")
        st.write(ACCIONES[arquetipo])
        if remove_rate == 0:
            st.caption("Esta sesion no edito el carrito - la senal de abandono mas fuerte.")

# ============================================================
# TAB 2 - LOTES (con carga de CSV)
# ============================================================
with tab2:
    st.subheader("Procesamiento por lotes")

    modo = st.radio("Origen de los datos:",
                    ["Muestra de validacion (demo)", "Subir mi propio CSV"])

    if modo == "Muestra de validacion (demo)":
        st.write("Muestra del conjunto de validacion (datos que el modelo NO vio "
                 "en entrenamiento).")
        n_muestra = st.select_slider("Tamano de la muestra",
                                     options=[100, 500, 1000, 5000], value=1000)
        if st.button("Procesar lote"):
            valid = cargar_validacion()
            muestra = valid.sample(n=min(n_muestra, len(valid)), random_state=1)
            resultado = procesar_lote(muestra)
            st.divider()
            mostrar_resultados(resultado)

    else:
        st.write("Sube un CSV con las sesiones de tu negocio. Debe contener las "
                 "columnas que el modelo necesita. Descarga la plantilla para ver "
                 "el formato exacto.")

        # Plantilla de ejemplo
        plantilla = pd.DataFrame([{
            "n_events": 8, "n_carts": 3, "n_categories": 3, "price_avg": 15.0,
            "price_max": 20.0, "price_std": 5.0, "brand_null_ratio": 0.0,
            "hour": 14, "dayofweek": 3, "is_weekend": 0, "remove_rate": 0.33,
            "products_per_event": 0.625, "brands_per_product": 0.4,
            "log_duration": 5.9, "log_events": 2.2,
        }])
        st.download_button("Descargar plantilla CSV",
                           plantilla.to_csv(index=False),
                           "plantilla_sesiones.csv", "text/csv")

        archivo = st.file_uploader("Sube tu archivo CSV", type=["csv"])
        if archivo is not None:
            try:
                datos = pd.read_csv(archivo)
                faltan = [c for c in XGB_FEATURES if c not in datos.columns]
                if faltan:
                    st.error(f"Al CSV le faltan columnas: {faltan}")
                else:
                    st.success(f"Archivo cargado: {len(datos):,} sesiones.")
                    resultado = procesar_lote(datos)
                    st.divider()
                    mostrar_resultados(resultado)
            except Exception as e:
                st.error(f"Error al leer el archivo: {e}")

st.divider()
st.caption("Umbral: 0.33 (optimo de negocio). Modelo 1: XGBoost - Modelo 2: K-Means.")