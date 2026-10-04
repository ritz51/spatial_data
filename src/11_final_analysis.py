from pyspark.sql import SparkSession
from pyspark.sql.functions import (
    col,
    unix_timestamp,
    hour,
    count,
    avg,
    min,
    max,
    percentile_approx,
    row_number
)
from pyspark.sql.window import Window


# ============================================================
# 1. SPARK SESSION
# ============================================================

spark = SparkSession.builder \
    .appName("NYC Taxi Final Spatial Data Analysis") \
    .master("local[*]") \
    .getOrCreate()

spark.sparkContext.setLogLevel("WARN")


# ============================================================
# 2. LOAD DATA
# ============================================================

taxi_file = r"C:\Users\Rithanya\Downloads\spaitial data\data\yellow_tripdata_2026-01.parquet"

df = spark.read.parquet(taxi_file)

original_count = df.count()

print("\n")
print("=" * 60)
print("       NYC TAXI SPATIAL DATA ANALYSIS")
print("=" * 60)

print("\nOriginal records:", original_count)


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
# 4. DATA CLEANING
# ============================================================

clean_df = df.filter(
    (col("trip_distance") > 0) &
    (col("trip_distance") <= 100) &
    (col("trip_duration_minutes") > 0) &
    (col("trip_duration_minutes") <= 300) &
    col("PULocationID").isNotNull() &
    col("DOLocationID").isNotNull()
)


# ============================================================
# 5. CALCULATE AVERAGE SPEED
# ============================================================

clean_df = clean_df.withColumn(
    "average_speed_mph",
    col("trip_distance") /
    (col("trip_duration_minutes") / 60)
)


# ============================================================
# 6. SPEED FILTER
# ============================================================

clean_df = clean_df.filter(
    (col("average_speed_mph") >= 1) &
    (col("average_speed_mph") <= 100)
)


final_count = clean_df.count()

removed_count = original_count - final_count

retention_rate = (
    final_count / original_count
) * 100


print("\n")
print("=" * 60)
print("DATA CLEANING")
print("=" * 60)

print("Original records :", original_count)
print("Final records    :", final_count)
print("Removed records  :", removed_count)
print("Retention rate   :", round(retention_rate, 2), "%")


# ============================================================
# 7. ADD PICKUP HOUR
# ============================================================

clean_df = clean_df.withColumn(
    "pickup_hour",
    hour("tpep_pickup_datetime")
)


# ============================================================
# 8. SPATIAL ANALYSIS
# ============================================================

pickup_hotspots = clean_df.groupBy(
    "PULocationID"
).agg(
    count("*").alias("pickup_count"),
    avg("trip_duration_minutes").alias("avg_duration"),
    avg("trip_distance").alias("avg_distance")
).orderBy(
    col("pickup_count").desc()
)


dropoff_hotspots = clean_df.groupBy(
    "DOLocationID"
).agg(
    count("*").alias("dropoff_count")
).orderBy(
    col("dropoff_count").desc()
)


top_pickup = pickup_hotspots.first()
top_dropoff = dropoff_hotspots.first()


print("\n")
print("=" * 60)
print("SPATIAL ANALYSIS")
print("=" * 60)

print(
    "Top pickup zone :",
    top_pickup["PULocationID"],
    "| Trips:",
    top_pickup["pickup_count"]
)

print(
    "Top drop-off zone:",
    top_dropoff["DOLocationID"],
    "| Trips:",
    top_dropoff["dropoff_count"]
)


print("\nTop 10 Pickup Zones:")

pickup_hotspots.show(10, truncate=False)


# ============================================================
# 9. TEMPORAL ANALYSIS
# ============================================================

hourly_demand = clean_df.groupBy(
    "pickup_hour"
).agg(
    count("*").alias("trip_count")
).orderBy(
    "pickup_hour"
)


peak_hour = hourly_demand.orderBy(
    col("trip_count").desc()
).first()


lowest_hour = hourly_demand.orderBy(
    col("trip_count").asc()
).first()


print("\n")
print("=" * 60)
print("TEMPORAL ANALYSIS")
print("=" * 60)

print(
    "Peak hour:",
    f"{peak_hour['pickup_hour']:02d}:00",
    "-",
    f"{peak_hour['pickup_hour'] + 1:02d}:00"
)

print(
    "Peak-hour trips:",
    peak_hour["trip_count"]
)

print(
    "Lowest-demand hour:",
    f"{lowest_hour['pickup_hour']:02d}:00"
)

print(
    "Lowest-hour trips:",
    lowest_hour["trip_count"]
)


print("\nHourly Demand:")

hourly_demand.show(24, truncate=False)


# ============================================================
# 10. TRIP DURATION ANALYSIS
# ============================================================

duration_stats = clean_df.select(
    avg("trip_duration_minutes").alias("average_duration"),
    percentile_approx(
        "trip_duration_minutes",
        0.5
    ).alias("median_duration"),
    min("trip_duration_minutes").alias("minimum_duration"),
    max("trip_duration_minutes").alias("maximum_duration")
).first()


print("\n")
print("=" * 60)
print("TRIP DURATION ANALYSIS")
print("=" * 60)

print(
    "Average duration:",
    round(duration_stats["average_duration"], 2),
    "minutes"
)

print(
    "Median duration :",
    round(float(duration_stats["median_duration"]), 2),
    "minutes"
)

print(
    "Minimum duration:",
    round(duration_stats["minimum_duration"], 2),
    "minutes"
)

print(
    "Maximum duration:",
    round(duration_stats["maximum_duration"], 2),
    "minutes"
)


# ============================================================
# 11. PEAK-HOUR SPATIAL ANALYSIS
# ============================================================

peak_hour_value = peak_hour["pickup_hour"]

peak_df = clean_df.filter(
    col("pickup_hour") == peak_hour_value
)


peak_pickups = peak_df.groupBy(
    "PULocationID"
).agg(
    count("*").alias("pickup_count"),
    avg("trip_duration_minutes").alias("avg_duration"),
    avg("trip_distance").alias("avg_distance")
).orderBy(
    col("pickup_count").desc()
)


peak_dropoffs = peak_df.groupBy(
    "DOLocationID"
).agg(
    count("*").alias("dropoff_count")
).orderBy(
    col("dropoff_count").desc()
)


peak_top_pickup = peak_pickups.first()
peak_top_dropoff = peak_dropoffs.first()


peak_duration = peak_df.select(
    avg("trip_duration_minutes").alias("avg_duration")
).first()


peak_distance = peak_df.select(
    avg("trip_distance").alias("avg_distance")
).first()


print("\n")
print("=" * 60)
print("PEAK-HOUR SPATIAL ANALYSIS")
print("=" * 60)

print(
    "Peak hour:",
    f"{peak_hour_value:02d}:00 - {peak_hour_value + 1:02d}:00"
)

print(
    "Peak-hour trips:",
    peak_df.count()
)

print(
    "Top pickup zone:",
    peak_top_pickup["PULocationID"],
    "| Trips:",
    peak_top_pickup["pickup_count"]
)

print(
    "Top drop-off zone:",
    peak_top_dropoff["DOLocationID"],
    "| Trips:",
    peak_top_dropoff["dropoff_count"]
)

print(
    "Average duration:",
    round(peak_duration["avg_duration"], 2),
    "minutes"
)

print(
    "Average distance:",
    round(peak_distance["avg_distance"], 2),
    "miles"
)


print("\nTop 10 Peak-Hour Pickup Zones:")

peak_pickups.show(10, truncate=False)


# ============================================================
# 12. ORIGIN → DESTINATION ANALYSIS
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


top_od = od_flows.first()


print("\n")
print("=" * 60)
print("ORIGIN → DESTINATION ANALYSIS")
print("=" * 60)

print(
    "Top OD pair:",
    top_od["PULocationID"],
    "→",
    top_od["DOLocationID"]
)

print(
    "Trips:",
    top_od["trip_count"]
)

print(
    "Average duration:",
    round(top_od["avg_duration"], 2),
    "minutes"
)

print(
    "Average distance:",
    round(top_od["avg_distance"], 2),
    "miles"
)


print("\nTop 10 OD Flows:")

od_flows.show(10, truncate=False)


# ============================================================
# 13. SAME-ZONE VS DIFFERENT-ZONE
# ============================================================

same_zone = clean_df.filter(
    col("PULocationID") == col("DOLocationID")
).count()


different_zone = clean_df.filter(
    col("PULocationID") != col("DOLocationID")
).count()


same_zone_percentage = (
    same_zone / final_count
) * 100


different_zone_percentage = (
    different_zone / final_count
) * 100


print("\n")
print("=" * 60)
print("ZONE MOVEMENT ANALYSIS")
print("=" * 60)

print(
    "Same-zone trips:",
    same_zone,
    "(",
    round(same_zone_percentage, 2),
    "%)"
)

print(
    "Different-zone trips:",
    different_zone,
    "(",
    round(different_zone_percentage, 2),
    "%)"
)


# ============================================================
# 14. FINAL SUMMARY
# ============================================================

print("\n")
print("=" * 60)
print("                FINAL SUMMARY")
print("=" * 60)

print("Original records       :", original_count)
print("Final cleaned records  :", final_count)

print(
    "Top pickup zone        :",
    top_pickup["PULocationID"]
)

print(
    "Top drop-off zone      :",
    top_dropoff["DOLocationID"]
)

print(
    "Peak operating hour    :",
    f"{peak_hour_value:02d}:00"
)

print(
    "Peak-hour trips        :",
    peak_hour["trip_count"]
)

print(
    "Average trip duration  :",
    round(duration_stats["average_duration"], 2),
    "minutes"
)

print(
    "Median trip duration   :",
    round(float(duration_stats["median_duration"]), 2),
    "minutes"
)

print(
    "Top OD flow            :",
    f"{top_od['PULocationID']} → {top_od['DOLocationID']}"
)

print(
    "Top OD flow trips      :",
    top_od["trip_count"]
)

print(
    "Same-zone trips        :",
    round(same_zone_percentage, 2),
    "%"
)

print(
    "Different-zone trips   :",
    round(different_zone_percentage, 2),
    "%"
)

print("\n")
print("=" * 60)
print("             ANALYSIS COMPLETED")
print("=" * 60)


# ============================================================
# 15. STOP SPARK
# ============================================================

spark.stop()