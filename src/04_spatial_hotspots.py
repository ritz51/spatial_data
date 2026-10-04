from pyspark.sql import SparkSession
from pyspark.sql.functions import (
    col,
    unix_timestamp,
    avg,
    count
)

import geopandas as gpd
import folium
import branca.colormap as cm


# ============================================================
# 1. CREATE SPARK SESSION
# ============================================================

spark = SparkSession.builder \
    .appName("NYC Taxi Spatial Hotspot Analysis") \
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
# ============================================================
# 4. CLEAN DATA
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
# 4.1 CALCULATE AVERAGE SPEED
# ============================================================

clean_df = clean_df.withColumn(
    "average_speed_mph",
    col("trip_distance") /
    (col("trip_duration_minutes") / 60)
)


# ============================================================
# 4.2 REMOVE UNREALISTIC SPEEDS
# ============================================================

clean_df = clean_df.filter(
    (col("average_speed_mph") >= 1) &
    (col("average_speed_mph") <= 100)
)


print("Records after cleaning:", clean_df.count())

# ============================================================
# 5. CREATE SPARK SQL VIEW
# ============================================================

clean_df.createOrReplaceTempView("taxi")


# ============================================================
# 6. SPATIAL HOTSPOT ANALYSIS
# ============================================================

pickup_hotspots = spark.sql("""
    SELECT
        PULocationID,
        COUNT(*) AS pickup_count,
        ROUND(AVG(trip_duration_minutes), 2) AS avg_duration,
        ROUND(AVG(trip_distance), 2) AS avg_distance
    FROM taxi
    GROUP BY PULocationID
    ORDER BY pickup_count DESC
""")


print("\n==============================================")
print("TOP PICKUP HOTSPOTS")
print("==============================================")

pickup_hotspots.show(20, truncate=False)


# ============================================================
# 7. CONVERT SPARK RESULT TO PANDAS
# ============================================================

pickup_pd = pickup_hotspots.toPandas()

pickup_pd["PULocationID"] = pickup_pd["PULocationID"].astype(int)


# ============================================================
# 8. LOAD NYC TAXI ZONE SHAPEFILE
# ============================================================

shapefile_path = r"C:\Users\Rithanya\Downloads\spaitial data\data\taxi_zones\taxi_zones.shp"

zones = gpd.read_file(shapefile_path)

print("\n==============================================")
print("TAXI ZONE INFORMATION")
print("==============================================")

print(zones.head())

print("\nColumns:")
print(zones.columns)


# ============================================================
# 9. CONVERT COORDINATE SYSTEM
# ============================================================

zones = zones.to_crs(epsg=4326)

zones["LocationID"] = zones["LocationID"].astype(int)


# ============================================================
# 10. MERGE TAXI DATA WITH TAXI ZONES
# ============================================================

hotspot_zones = zones.merge(
    pickup_pd,
    left_on="LocationID",
    right_on="PULocationID",
    how="left"
)


# Zones with no trips get 0
hotspot_zones["pickup_count"] = (
    hotspot_zones["pickup_count"]
    .fillna(0)
)


hotspot_zones["avg_duration"] = (
    hotspot_zones["avg_duration"]
    .fillna(0)
)


hotspot_zones["avg_distance"] = (
    hotspot_zones["avg_distance"]
    .fillna(0)
)


# ============================================================
# 11. CREATE NYC MAP
# ============================================================

m = folium.Map(
    location=[40.7128, -74.0060],
    zoom_start=11,
    tiles="CartoDB positron"
)


# ============================================================
# 12. CREATE COLOR SCALE
# ============================================================

max_pickups = hotspot_zones["pickup_count"].max()

color_scale = cm.LinearColormap(
    colors=["yellow", "orange", "red"],
    vmin=0,
    vmax=max_pickups
)

color_scale.caption = "Number of Pickup Trips"

color_scale.add_to(m)


# ============================================================
# 13. FUNCTION FOR ZONE COLOR
# ============================================================

def get_color(pickup_count):

    return color_scale(pickup_count)


# ============================================================
# 14. ADD TAXI ZONES TO MAP
# ============================================================

folium.GeoJson(
    hotspot_zones.to_json(),

    name="NYC Taxi Pickup Hotspots",

    style_function=lambda feature: {
        "fillColor": get_color(
            feature["properties"]["pickup_count"]
        ),
        "color": "black",
        "weight": 0.5,
        "fillOpacity": 0.65
    },

    highlight_function=lambda feature: {
        "weight": 2,
        "color": "blue",
        "fillOpacity": 0.8
    },

    tooltip=folium.GeoJsonTooltip(
        fields=[
            "zone",
            "borough",
            "pickup_count",
            "avg_duration",
            "avg_distance"
        ],

        aliases=[
            "Taxi Zone:",
            "Borough:",
            "Pickup Count:",
            "Average Duration (min):",
            "Average Distance (miles):"
        ],

        localize=True,

        sticky=False
    )
).add_to(m)


# ============================================================
# 15. ADD LAYER CONTROL
# ============================================================

folium.LayerControl().add_to(m)


# ============================================================
# 16. SAVE MAP
# ============================================================

output_file = r"C:\Users\Rithanya\Downloads\spaitial data\maps\nyc_pickup_hotspots.html"

m.save(output_file)


print("\n==============================================")
print("SPATIAL HOTSPOT MAP CREATED SUCCESSFULLY")
print("==============================================")

print("\nMap location:")
print(output_file)


# ============================================================
# 17. STOP SPARK
# ============================================================

spark.stop()