from pyspark.sql import SparkSession
from pyspark.sql.functions import (
    col,
    unix_timestamp,
    hour,
    count,
    avg
)


# ============================================================
# 1. CREATE SPARK SESSION
# ============================================================

spark = SparkSession.builder \
    .appName("NYC Peak Hour Spatial Hotspot Analysis") \
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
# 3. CALCULATE TRIP DURATION
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


# ============================================================
# 5. EXTRACT PICKUP HOUR
# ============================================================

clean_df = clean_df.withColumn(
    "pickup_hour",
    hour("tpep_pickup_datetime")
)


# ============================================================
# 6. FILTER PEAK HOUR
# ============================================================

PEAK_HOUR = 18

peak_df = clean_df.filter(
    col("pickup_hour") == PEAK_HOUR
)


print("\n==============================================")
print("PEAK HOUR ANALYSIS")
print("==============================================")

print("Peak Hour: 18:00 - 19:00")
print("Total trips during peak hour:", peak_df.count())


# ============================================================
# 7. FIND TOP PICKUP ZONES DURING PEAK HOUR
# ============================================================

peak_hotspots = peak_df.groupBy(
    "PULocationID"
).agg(
    count("*").alias("pickup_count"),
    avg("trip_duration_minutes").alias("avg_duration"),
    avg("trip_distance").alias("avg_distance")
).orderBy(
    col("pickup_count").desc()
)


print("\n==============================================")
print("TOP PICKUP HOTSPOTS DURING 6 PM")
print("==============================================")

peak_hotspots.show(20, truncate=False)


# ============================================================
# 8. FIND TOP DROPOFF ZONES DURING PEAK HOUR
# ============================================================

peak_dropoffs = peak_df.groupBy(
    "DOLocationID"
).agg(
    count("*").alias("dropoff_count")
).orderBy(
    col("dropoff_count").desc()
)


print("\n==============================================")
print("TOP DROPOFF HOTSPOTS DURING 6 PM")
print("==============================================")

peak_dropoffs.show(20, truncate=False)


# ============================================================
# 9. FIND PEAK-HOUR AVERAGE TRIP DURATION
# ============================================================

peak_duration = peak_df.select(
    avg("trip_duration_minutes").alias("average_duration")
).first()


print("\n==============================================")
print("PEAK-HOUR TRIP DURATION")
print("==============================================")

print(
    "Average trip duration at 6 PM:",
    round(peak_duration["average_duration"], 2),
    "minutes"
)


# ============================================================
# 10. FIND PEAK-HOUR AVERAGE DISTANCE
# ============================================================

peak_distance = peak_df.select(
    avg("trip_distance").alias("average_distance")
).first()


print(
    "Average trip distance at 6 PM:",
    round(peak_distance["average_distance"], 2),
    "miles"
)


# ============================================================
# 11. STOP SPARK
# ============================================================

spark.stop()