from pyspark.sql import SparkSession
from pyspark.sql.functions import (
    col,
    unix_timestamp,
    hour,
    avg,
    min,
    max,
    percentile_approx,
    count
)


# ============================================================
# 1. CREATE SPARK SESSION
# ============================================================

spark = SparkSession.builder \
    .appName("NYC Taxi Trip Duration Analysis") \
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


print("Records after cleaning:", clean_df.count())


# ============================================================
# 5. EXTRACT PICKUP HOUR
# ============================================================

clean_df = clean_df.withColumn(
    "pickup_hour",
    hour("tpep_pickup_datetime")
)


# ============================================================
# 6. OVERALL TRIP DURATION STATISTICS
# ============================================================

duration_stats = clean_df.select(
    count("*").alias("total_trips"),
    avg("trip_duration_minutes").alias("average_duration"),
    min("trip_duration_minutes").alias("minimum_duration"),
    max("trip_duration_minutes").alias("maximum_duration"),
    percentile_approx(
        "trip_duration_minutes",
        0.5
    ).alias("median_duration")
)


print("\n==============================================")
print("OVERALL TRIP DURATION STATISTICS")
print("==============================================")

duration_stats.show(truncate=False)


# ============================================================
# 7. AVERAGE DURATION BY HOUR
# ============================================================

duration_by_hour = clean_df.groupBy(
    "pickup_hour"
).agg(
    count("*").alias("trip_count"),
    avg("trip_duration_minutes").alias("average_duration")
).orderBy(
    "pickup_hour"
)


print("\n==============================================")
print("AVERAGE TRIP DURATION BY HOUR")
print("==============================================")

duration_by_hour.show(24, truncate=False)


# ============================================================
# 8. LONGEST AVERAGE DURATION HOURS
# ============================================================

longest_duration_hours = duration_by_hour.orderBy(
    col("average_duration").desc()
)


print("\n==============================================")
print("HOURS WITH LONGEST AVERAGE TRIP DURATION")
print("==============================================")

longest_duration_hours.show(10, truncate=False)


# ============================================================
# 9. SHORTEST AVERAGE DURATION HOURS
# ============================================================

shortest_duration_hours = duration_by_hour.orderBy(
    col("average_duration").asc()
)


print("\n==============================================")
print("HOURS WITH SHORTEST AVERAGE TRIP DURATION")
print("==============================================")

shortest_duration_hours.show(10, truncate=False)


# ============================================================
# 10. AVERAGE DURATION BY PICKUP ZONE
# ============================================================

duration_by_zone = clean_df.groupBy(
    "PULocationID"
).agg(
    count("*").alias("trip_count"),
    avg("trip_duration_minutes").alias("average_duration"),
    avg("trip_distance").alias("average_distance")
).filter(
    col("trip_count") >= 100
).orderBy(
    col("average_duration").desc()
)


print("\n==============================================")
print("ZONES WITH LONGEST AVERAGE TRIP DURATION")
print("==============================================")

duration_by_zone.show(20, truncate=False)


# ============================================================
# 11. CORRELATION BETWEEN DISTANCE AND DURATION
# ============================================================

correlation = clean_df.stat.corr(
    "trip_distance",
    "trip_duration_minutes"
)


print("\n==============================================")
print("DISTANCE VS DURATION")
print("==============================================")

print(
    "Correlation between trip distance and duration:",
    round(correlation, 4)
)


# ============================================================
# 12. STOP SPARK
# ============================================================

spark.stop()

