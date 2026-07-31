"""
Sesionización con truncado en la primera compra.

Regla: para sesiones que terminan en compra, sólo se usan los eventos
ANTERIORES al primer 'purchase'. Esto evita fuga de información: el
modelo nunca ve el evento que define la etiqueta.
"""
from pyspark.sql import SparkSession, Window
from pyspark.sql import functions as F

TZ_OFFSET_HOURS = 5  # event_time en UTC; hora local del negocio = UTC+5

spark = (
    SparkSession.builder
    .master("local[*]")
    .appName("tfm-sessionize")
    .config("spark.driver.memory", "8g")
    .config("spark.sql.shuffle.partitions", "64")
    .getOrCreate()
)
spark.sparkContext.setLogLevel("WARN")

df = spark.read.csv("data/raw/*.csv", header=True, inferSchema=True)

# 1. Descartar eventos sin sesión (4.598 filas, 0,02%)
df = df.filter(F.col("user_session").isNotNull())

# 2. Etiqueta y timestamp de la primera compra, por sesión
w = Window.partitionBy("user_session")
df = df.withColumn(
    "first_purchase_ts",
    F.min(F.when(F.col("event_type") == "purchase", F.col("event_time"))).over(w),
)
df = df.withColumn("label", F.col("first_purchase_ts").isNotNull().cast("int"))

# 3. TRUNCADO: conservar sólo eventos previos a la primera compra
pre = df.filter(
    (F.col("first_purchase_ts").isNull())
    | (F.col("event_time") < F.col("first_purchase_ts"))
)

# 4. Hora local para features temporales interpretables
pre = pre.withColumn(
    "event_time_local", F.col("event_time") + F.expr(f"INTERVAL {TZ_OFFSET_HOURS} HOURS")
)

# 5. Agregación a nivel de sesión
sessions = pre.groupBy("user_session").agg(
    F.first("user_id").alias("user_id"),
    F.first("label").alias("label"),

    # Volumen e intensidad
    F.count("*").alias("n_events"),
    F.sum((F.col("event_type") == "view").cast("int")).alias("n_views"),
    F.sum((F.col("event_type") == "cart").cast("int")).alias("n_carts"),
    F.sum((F.col("event_type") == "remove_from_cart").cast("int")).alias("n_removes"),

    # Amplitud de exploración
    F.countDistinct("product_id").alias("n_products"),
    F.countDistinct("brand").alias("n_brands"),
    F.countDistinct("category_id").alias("n_categories"),

    # Precio
    F.avg("price").alias("price_avg"),
    F.max("price").alias("price_max"),
    F.min("price").alias("price_min"),
    F.stddev("price").alias("price_std"),

    # Temporalidad
    F.min("event_time").alias("session_start"),
    F.max("event_time").alias("session_end"),
    F.first("event_time_local").alias("first_event_local"),

    # Calidad de dato
    F.avg(F.col("brand").isNull().cast("int")).alias("brand_null_ratio"),
)

# 6. Features derivadas
sessions = (
    sessions
    .withColumn(
        "duration_sec",
        F.col("session_end").cast("long") - F.col("session_start").cast("long"),
    )
    .withColumn("hour", F.hour("first_event_local"))
    .withColumn("dayofweek", F.dayofweek("first_event_local"))
    .withColumn("is_weekend", F.col("dayofweek").isin([1, 7]).cast("int"))
    .withColumn("cart_rate", F.col("n_carts") / F.col("n_events"))
    .withColumn(
        "remove_rate",
        F.when(F.col("n_carts") > 0, F.col("n_removes") / F.col("n_carts")).otherwise(0.0),
    )
    .withColumn(
        "events_per_min",
        F.when(F.col("duration_sec") > 0, F.col("n_events") * 60 / F.col("duration_sec"))
        .otherwise(F.col("n_events").cast("double")),
    )
    .withColumn("price_std", F.coalesce(F.col("price_std"), F.lit(0.0)))
    .drop("first_event_local")
)

# 7. Diagnóstico
total = sessions.count()
positivos = sessions.filter(F.col("label") == 1).count()
print(f"\nSesiones tras truncado: {total:,}")
print(f"Sesiones con compra:    {positivos:,} ({100*positivos/total:.2f}%)")

sessions.write.mode("overwrite").parquet("data/processed/sessions.parquet")
print("\nGuardado en data/processed/sessions.parquet")

spark.stop()