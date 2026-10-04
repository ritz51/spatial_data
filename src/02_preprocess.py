from pyspark.sql import SparkSession
from pyspark.sql.functions import (
    col,
    hour,
    dayofweek,
    datediff,
    unix_timestamp
)

# --------------------------------------------------
# 1. Start Spark
# --------------------------------------------------

spark = SparkSession.builder \
    .appName("NYC Taxi Spatial Analysis") \
    .master("local[*]") \
    .getOrCreate()

# --------------------------------------------------
# 2. Load January 2026 data
# --------------------------------------------------

df =  spark.read.parquet(
    r"C:\Users\Rithanya\Downloads\spaitial data\data\yellow_tripdata_2026-01.parquet"
)

print("\n========== ORIGINAL RECORD COUNT ==========")
print(df.count())

# --------------------------------------------------
# 3. Select required columns
# --------------------------------------------------

df = df.select(
    "VendorID",
    "tpep_pickup_datetime",
    "tpep_dropoff_datetime",
    "passenger_count",
    "trip_distance",
    "PULocationID",
    "DOLocationID",
    "fare_amount",
    "total_amount"
)

# --------------------------------------------------
# 4. Calculate trip duration
# --------------------------------------------------

df = df.withColumn(
    "trip_duration_minutes",
    (
        unix_timestamp("tpep_dropoff_datetime")
        - unix_timestamp("tpep_pickup_datetime")
    ) / 60
)

# --------------------------------------------------
# 5. Extract pickup hour
# --------------------------------------------------

df = df.withColumn(
    "pickup_hour",
    hour("tpep_pickup_datetime")
)

# --------------------------------------------------
# 6. Extract day of week
# --------------------------------------------------

df = df.withColumn(
    "day_of_week",
    dayofweek("tpep_pickup_datetime")
)

# --------------------------------------------------
# 7. Remove invalid records
# --------------------------------------------------

df_clean = df.filter(
    (col("trip_distance") > 0) &
    (col("trip_duration_minutes") > 0) &
    (col("trip_duration_minutes") <= 300) &
    (col("PULocationID").isNotNull()) &
    (col("DOLocationID").isNotNull())
)

print("\n========== CLEAN RECORD COUNT ==========")
print(df_clean.count())

# --------------------------------------------------
# 8. Display processed data
# --------------------------------------------------

print("\n========== PROCESSED DATA ==========")

df_clean.select(
    "tpep_pickup_datetime",
    "tpep_dropoff_datetime",
    "PULocationID",
    "DOLocationID",
    "trip_distance",
    "trip_duration_minutes",
    "pickup_hour",
    "day_of_week"
).show(10, truncate=False)

# --------------------------------------------------
# 9. Save cleaned data
# --------------------------------------------------
print("\nData preprocessing completed successfully!")

# --------------------------------------------------
# 10. Stop Spark
# --------------------------------------------------

spark.stop()