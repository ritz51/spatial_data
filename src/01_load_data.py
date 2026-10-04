from pyspark.sql import SparkSession

# Create Spark session
spark = SparkSession.builder \
    .appName("NYC Taxi Spatial Analysis") \
    .master("local[*]") \
    .getOrCreate()

# Load January 2026 NYC Yellow Taxi data
df =  spark.read.parquet(
    r"C:\Users\Rithanya\Downloads\spaitial data\data\yellow_tripdata_2026-01.parquet"
)

# Display schema
print("\n========== DATASET SCHEMA ==========")
df.printSchema()

# Count records
print("\n========== TOTAL RECORDS ==========")
print("Number of rows:", df.count())

# Display first 5 records
print("\n========== SAMPLE RECORDS ==========")
df.show(5, truncate=False)

# Stop Spark
spark.stop()