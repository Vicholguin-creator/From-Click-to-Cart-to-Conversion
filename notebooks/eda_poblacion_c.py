"""EDA de la población C: sesiones con >=1 cart."""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

sns.set_theme(style="whitegrid", palette="muted")
pd.set_option("display.float_format", "{:.2f}".format)

df = pd.read_parquet("data/processed/sessions.parquet")
c = df[df.n_carts >= 1].copy()
print(f"Población C: {len(c):,} sesiones | conversión: {c.label.mean():.2%}\n")

# ---------- 1. Outliers temporales ----------
c["duration_min"] = c.duration_sec / 60
print("Duración (minutos):")
print(c.duration_min.describe(percentiles=[.5, .9, .95, .99, .999]), "\n")

largas = (c.duration_min > 24 * 60).sum()
print(f"Sesiones > 24h: {largas:,} ({100*largas/len(c):.2f}%)")
cero = (c.duration_sec == 0).sum()
print(f"Sesiones de 0 seg: {cero:,} ({100*cero/len(c):.2f}%)\n")

# ---------- 2. Distribuciones clave por clase ----------
fig, axes = plt.subplots(2, 3, figsize=(16, 9))
feats = ["n_events", "n_carts", "n_products", "duration_min", "remove_rate", "price_avg"]

for ax, f in zip(axes.flat, feats):
    for lab, color in [(0, "#c44"), (1, "#4a8")]:
        data = c.loc[c.label == lab, f]
        data = data[data <= data.quantile(0.99)]   # recorte visual
        ax.hist(data, bins=50, alpha=.55, label=f"label={lab}",
                color=color, density=True)
    ax.set_title(f)
    ax.legend()
plt.tight_layout()
plt.savefig("notebooks/dist_por_clase.png", dpi=110)
print("→ notebooks/dist_por_clase.png\n")

# ---------- 3. remove_rate: la cola vacilante ----------
print("remove_rate por clase:")
print(c.groupby("label").remove_rate.describe(percentiles=[.5, .75, .9, .99]), "\n")
sin_remove = (c.n_removes == 0).groupby(c.label).mean()
print("Proporción sin ningún remove:")
print(sin_remove, "\n")

# ---------- 4. Tasa de conversión por tramo ----------
c["ev_bin"] = pd.cut(c.n_events, [0, 1, 2, 5, 10, 20, 50, np.inf])
conv = c.groupby("ev_bin", observed=True).agg(
    sesiones=("label", "size"), conversion=("label", "mean")
)
print("Conversión por nº de eventos:")
print(conv, "\n")

fig, ax = plt.subplots(figsize=(9, 5))
conv.conversion.mul(100).plot(kind="bar", ax=ax, color="#4a8")
ax.set_ylabel("% conversión"); ax.set_xlabel("nº eventos en sesión")
ax.set_title("Conversión por intensidad de sesión — población C")
plt.tight_layout()
plt.savefig("notebooks/conv_por_eventos.png", dpi=110)
print("→ notebooks/conv_por_eventos.png\n")

# ---------- 5. Correlaciones ----------
num = ["n_events", "n_views", "n_carts", "n_removes", "n_products", "n_brands",
       "duration_sec", "cart_rate", "remove_rate", "events_per_min",
       "price_avg", "price_std", "hour", "is_weekend", "label"]
corr = c[num].corr(method="spearman")

fig, ax = plt.subplots(figsize=(11, 9))
sns.heatmap(corr, annot=True, fmt=".2f", cmap="RdBu_r", center=0,
            square=True, cbar_kws={"shrink": .7}, ax=ax)
ax.set_title("Correlación de Spearman — población C")
plt.tight_layout()
plt.savefig("notebooks/corr_spearman.png", dpi=110)
print("→ notebooks/corr_spearman.png\n")

print("Correlación con label (ordenada):")
print(corr.label.drop("label").sort_values(ascending=False))