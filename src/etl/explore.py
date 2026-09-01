"""Primer vistazo al dataset REES46 de cosmética."""
from pyspark.sql import SparkSession
from pyspark.sql import functions as F

spark = (
    SparkSession.builder
    .master("local[*]")
    .appName("tfm-explore")
    .config("spark.driver.memory", "8g")
    .getOrCreate()
)
spark.sparkContext.setLogLevel("WARN")

df = spark.read.csv("data/raw/*.csv", header=True, inferSchema=True)

print("\n--- ESQUEMA ---")
df.printSchema()

print(f"\n--- VOLUMEN ---")
print(f"Eventos totales: {df.count():,}")

print("\n--- TIPOS DE EVENTO ---")
df.groupBy("event_type").count().orderBy(F.desc("count")).show()

print("\n--- RANGO TEMPORAL ---")
df.select(F.min("event_time"), F.max("event_time")).show(truncate=False)

print("\n--- NULOS POR COLUMNA ---")
df.select([
    F.sum(F.col(c).isNull().cast("int")).alias(c) for c in df.columns
]).show()

print("\n--- CARDINALIDAD ---")
print(f"Usuarios únicos:  {df.select('user_id').distinct().count():,}")
print(f"Sesiones únicas:  {df.select('user_session').distinct().count():,}")
print(f"Productos únicos: {df.select('product_id').distinct().count():,}")

spark.stop()
