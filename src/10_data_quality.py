from pyspark.sql import SparkSession
from pyspark.sql.functions import (
    col,
    unix_timestamp,
    round as spark_round
)


# ============================================================
# 1. CREATE SPARK SESSION
# ============================================================

spark = SparkSession.builder \
    .appName("NYC Taxi Data Quality Analysis") \
    .master("local[*]") \
    .getOrCreate()

spark.sparkContext.setLogLevel("WARN")


# ============================================================
# 2. LOAD DATA
# ============================================================

taxi_file = r"C:\Users\Rithanya\Downloads\spaitial data\data\yellow_tripdata_2026-01.parquet"

df = spark.read.parquet(taxi_file)

original_count = df.count()

print("\n==============================================")
print("DATA QUALITY ANALYSIS")
print("==============================================")

print("Original records:", original_count)


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
# 4. INITIAL CLEANING
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
# 6. REMOVE UNREALISTIC SPEEDS
# ============================================================

final_df = clean_df.filter(
    (col("average_speed_mph") >= 1) &
    (col("average_speed_mph") <= 100)
)


final_count = final_df.count()


# ============================================================
# 7. RECORD SUMMARY
# ============================================================

removed = original_count - final_count

retention = (
    final_count / original_count
) * 100


print("\n==============================================")
print("CLEANING SUMMARY")
print("==============================================")

print("Original records :", original_count)
print("Final records    :", final_count)
print("Removed records  :", removed)

print(
    "Retention rate   :",
    round(retention, 2),
    "%"
)


# ============================================================
# 8. CHECK FINAL DATA
# ============================================================

print("\n==============================================")
print("FINAL DATA SAMPLE")
print("==============================================")

final_df.select(
    "PULocationID",
    "DOLocationID",
    "trip_distance",
    "trip_duration_minutes",
    "average_speed_mph"
).show(10, truncate=False)


# ============================================================
# 9. CHECK MAXIMUM VALUES
# ============================================================

print("\n==============================================")
print("FINAL DATA LIMITS")
print("==============================================")

final_df.select(
    spark_round(
        col("trip_distance"),
        2
    ).alias("distance"),
    spark_round(
        col("trip_duration_minutes"),
        2
    ).alias("duration"),
    spark_round(
        col("average_speed_mph"),
        2
    ).alias("speed")
).orderBy(
    col("distance").desc()
).show(10)


# ============================================================
# 10. STOP SPARK
# ============================================================

spark.stop()