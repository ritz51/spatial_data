from pyspark.sql import SparkSession
from pyspark.sql.functions import (
    col,
    hour,
    dayofweek,
    unix_timestamp
)

# --------------------------------------------------
# 1. Start Spark
# --------------------------------------------------

spark = SparkSession.builder \
    .appName("NYC Taxi Hotspot Analysis") \
    .master("local[*]") \
    .getOrCreate()

# --------------------------------------------------
# 2. Load January 2026 data
# --------------------------------------------------

df =  spark.read.parquet(
    r"C:\Users\Rithanya\Downloads\spaitial data\data\yellow_tripdata_2026-01.parquet"
)
print("\n========== ORIGINAL RECORDS ==========")
print(df.count())

# --------------------------------------------------
# 3. Create trip duration
# --------------------------------------------------

df = df.withColumn(
    "trip_duration_minutes",
    (
        unix_timestamp("tpep_dropoff_datetime")
        - unix_timestamp("tpep_pickup_datetime")
    ) / 60
)

# --------------------------------------------------
# 4. Create pickup hour
# --------------------------------------------------

df = df.withColumn(
    "pickup_hour",
    hour("tpep_pickup_datetime")
)

# --------------------------------------------------
# 5. Clean data
# --------------------------------------------------

df_clean = df.filter(
    (col("trip_distance") > 0) &
    (col("trip_duration_minutes") > 0) &
    (col("trip_duration_minutes") <= 300) &
    (col("PULocationID").isNotNull()) &
    (col("DOLocationID").isNotNull())
)

print("\n========== CLEAN RECORDS ==========")
print(df_clean.count())

# --------------------------------------------------
# 6. Create temporary SQL table
# --------------------------------------------------

df_clean.createOrReplaceTempView("taxi")

# --------------------------------------------------
# 7. TOP 10 PICKUP HOTSPOTS
# --------------------------------------------------

print("\n========== TOP 10 PICKUP HOTSPOTS ==========")

top_pickup = spark.sql("""
    SELECT
        PULocationID,
        COUNT(*) AS pickup_count
    FROM taxi
    GROUP BY PULocationID
    ORDER BY pickup_count DESC
    LIMIT 10
""")

top_pickup.show()

# --------------------------------------------------
# 8. TOP 10 DROP-OFF HOTSPOTS
# --------------------------------------------------

print("\n========== TOP 10 DROP-OFF HOTSPOTS ==========")

top_dropoff = spark.sql("""
    SELECT
        DOLocationID,
        COUNT(*) AS dropoff_count
    FROM taxi
    GROUP BY DOLocationID
    ORDER BY dropoff_count DESC
    LIMIT 10
""")

top_dropoff.show()

# --------------------------------------------------
# 9. HOURLY TAXI DEMAND
# --------------------------------------------------

print("\n========== HOURLY TAXI DEMAND ==========")

hourly_demand = spark.sql("""
    SELECT
        pickup_hour,
        COUNT(*) AS trip_count
    FROM taxi
    GROUP BY pickup_hour
    ORDER BY pickup_hour
""")

hourly_demand.show(24)

# --------------------------------------------------
# 10. TOP HOTSPOT FOR EACH HOUR
# --------------------------------------------------

print("\n========== HOTSPOTS BY HOUR ==========")

hourly_hotspots = spark.sql("""
    SELECT
        pickup_hour,
        PULocationID,
        COUNT(*) AS pickup_count
    FROM taxi
    GROUP BY pickup_hour, PULocationID
    ORDER BY pickup_hour, pickup_count DESC
""")

hourly_hotspots.show(50)

# --------------------------------------------------
# Stop Spark
# --------------------------------------------------

spark.stop()