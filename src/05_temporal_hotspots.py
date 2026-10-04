from pyspark.sql import SparkSession
from pyspark.sql.functions import (
    col,
    unix_timestamp,
    hour,
    count,
    row_number
)


from pyspark.sql.window import Window


# ============================================================
# 1. CREATE SPARK SESSION
# ============================================================

spark = SparkSession.builder \
    .appName("NYC Taxi Temporal Hotspot Analysis") \
    .master("local[*]") \
    .getOrCreate()

spark.sparkContext.setLogLevel("WARN")


# ============================================================
# 2. LOAD TAXI DATA
# ============================================================

taxi_file = r"C:\Users\Rithanya\Downloads\spaitial data\data\yellow_tripdata_2026-01.parquet"

df = spark.read.parquet(taxi_file)

print("\nOriginal records:", df.count())


# ============================================================
# 3. CREATE TRIP DURATION
# ============================================================

df = df.withColumn(
    "trip_duration_minutes",
    (
        unix_timestamp("tpep_dropoff_datetime")
        - unix_timestamp("tpep_pickup_datetime")
    ) / 60
)


# ============================================================
# 4. CLEAN DATA
# ============================================================

clean_df = df.filter(
    (col("trip_distance") > 0) &
    (col("trip_duration_minutes") > 0) &
    (col("trip_duration_minutes") <= 300) &
    col("PULocationID").isNotNull() &
    col("DOLocationID").isNotNull()
)

print("Records after cleaning:", clean_df.count())


# ============================================================
# 5. EXTRACT PICKUP HOUR
# ============================================================

clean_df = clean_df.withColumn(
    "pickup_hour",
    hour("tpep_pickup_datetime")
)


# ============================================================
# 6. HOURLY DEMAND
# ============================================================

hourly_demand = clean_df.groupBy(
    "pickup_hour"
).agg(
    count("*").alias("trip_count")
).orderBy(
    "pickup_hour"
)


print("\n==============================================")
print("HOURLY TAXI DEMAND")
print("==============================================")

hourly_demand.show(24, truncate=False)


# ============================================================
# 7. FIND TOP PICKUP ZONES FOR EACH HOUR
# ============================================================

hourly_zone_demand = clean_df.groupBy(
    "pickup_hour",
    "PULocationID"
).agg(
    count("*").alias("pickup_count")
)


# ============================================================
# 8. RANK ZONES WITHIN EACH HOUR
# ============================================================

window_spec = Window.partitionBy(
    "pickup_hour"
).orderBy(
    col("pickup_count").desc()
)


ranked_zones = hourly_zone_demand.withColumn(
    "rank",
    row_number().over(window_spec)
)


# ============================================================
# 9. SELECT TOP 10 ZONES FOR EACH HOUR
# ============================================================

top10_hourly = ranked_zones.filter(
    col("rank") <= 10
).orderBy(
    "pickup_hour",
    "rank"
)


print("\n==============================================")
print("TOP 10 PICKUP ZONES FOR EACH HOUR")
print("==============================================")

top10_hourly.show(240, truncate=False)


# ============================================================
# 10. FIND PEAK HOUR
# ============================================================

peak_hour = hourly_demand.orderBy(
    col("trip_count").desc()
).first()


print("\n==============================================")
print("PEAK OPERATING HOUR")
print("==============================================")

print(
    "Peak Hour:",
    peak_hour["pickup_hour"],
    ":00"
)

print(
    "Number of Trips:",
    peak_hour["trip_count"]
)


# ============================================================
# 11. FIND LOWEST DEMAND HOUR
# ============================================================

lowest_hour = hourly_demand.orderBy(
    col("trip_count").asc()
).first()


print("\n==============================================")
print("LOWEST DEMAND HOUR")
print("==============================================")

print(
    "Lowest Hour:",
    lowest_hour["pickup_hour"],
    ":00"
)

print(
    "Number of Trips:",
    lowest_hour["trip_count"]
)


# ============================================================
# 12. STOP SPARK
# ============================================================

spark.stop()