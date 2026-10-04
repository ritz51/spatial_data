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
    .appName("NYC Taxi Origin Destination Flow Analysis") \
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
# 5. ORIGIN → DESTINATION FLOW
# ============================================================

od_flows = clean_df.groupBy(
    "PULocationID",
    "DOLocationID"
).agg(
    count("*").alias("trip_count"),
    avg("trip_duration_minutes").alias("avg_duration"),
    avg("trip_distance").alias("avg_distance")
).orderBy(
    col("trip_count").desc()
)


# ============================================================
# 6. TOP 20 OD FLOWS
# ============================================================

print("\n==============================================")
print("TOP 20 ORIGIN → DESTINATION FLOWS")
print("==============================================")

od_flows.show(20, truncate=False)


# ============================================================
# 7. PEAK-HOUR OD FLOWS
# ============================================================

clean_df = clean_df.withColumn(
    "pickup_hour",
    hour("tpep_pickup_datetime")
)

peak_od_flows = clean_df.filter(
    col("pickup_hour") == 18
).groupBy(
    "PULocationID",
    "DOLocationID"
).agg(
    count("*").alias("trip_count"),
    avg("trip_duration_minutes").alias("avg_duration"),
    avg("trip_distance").alias("avg_distance")
).orderBy(
    col("trip_count").desc()
)


print("\n==============================================")
print("TOP 20 ORIGIN → DESTINATION FLOWS AT 6 PM")
print("==============================================")

peak_od_flows.show(20, truncate=False)


# ============================================================
# 8. SAME-ZONE TRIPS
# ============================================================

same_zone = clean_df.filter(
    col("PULocationID") == col("DOLocationID")
).count()


total_trips = clean_df.count()


same_zone_percentage = (
    same_zone / total_trips
) * 100


print("\n==============================================")
print("SAME-ZONE TRIPS")
print("==============================================")

print("Same-zone trips:", same_zone)

print(
    "Percentage of same-zone trips:",
    round(same_zone_percentage, 2),
    "%"
)


# ============================================================
# 9. DIFFERENT-ZONE TRIPS
# ============================================================

different_zone = clean_df.filter(
    col("PULocationID") != col("DOLocationID")
).count()


different_zone_percentage = (
    different_zone / total_trips
) * 100


print("\n==============================================")
print("DIFFERENT-ZONE TRIPS")
print("==============================================")

print("Different-zone trips:", different_zone)

print(
    "Percentage of different-zone trips:",
    round(different_zone_percentage, 2),
    "%"
)


# ============================================================
# 10. STOP SPARK
# ============================================================

spark.stop()


