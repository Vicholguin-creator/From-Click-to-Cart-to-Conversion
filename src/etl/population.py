"""Comparación de poblaciones candidatas para el modelado."""
from pyspark.sql import SparkSession
from pyspark.sql import functions as F

spark = (
    SparkSession.builder.master("local[*]")
    .appName("tfm-population")
    .config("spark.driver.memory", "8g")
    .getOrCreate()
)
spark.sparkContext.setLogLevel("ERROR")

s = spark.read.parquet("data/processed/sessions.parquet")

def resumen(df, nombre):
    total = df.count()
    pos = df.filter(F.col("label") == 1).count()
    tasa = 100 * pos / total if total else 0
    ratio = (total - pos) / pos if pos else float("inf")
    print(f"\n{nombre}")
    print(f"  Sesiones:  {total:>10,}")
    print(f"  Positivas: {pos:>10,}  ({tasa:.2f}%)")
    print(f"  Desbalance: 1:{ratio:.0f}")

resumen(s, "A) Población completa")
resumen(s.filter(F.col("n_events") >= 2), "B) Sesiones con >=2 eventos")
resumen(s.filter(F.col("n_carts") >= 1), "C) Sesiones con >=1 cart (intención manifestada)")

# Cuánto tráfico es no accionable
un_evento = s.filter(F.col("n_events") == 1).count()
total = s.count()
print(f"\nSesiones de un solo evento: {un_evento:,} ({100*un_evento/total:.1f}% del total)")

# Dentro de C: cómo se ve el abandono
c = s.filter(F.col("n_carts") >= 1)
print("\n--- Dentro de la población C ---")
c.select("n_events", "n_carts", "n_removes", "remove_rate", "duration_sec") \
 .summary("min", "25%", "50%", "75%", "max").show()

spark.stop()