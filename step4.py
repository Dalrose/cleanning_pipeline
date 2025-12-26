from pyspark.sql import SparkSession
from pyspark.sql.functions import col, lit, when, rand
from pyspark.sql.functions import count as spark_count

spark = SparkSession.builder \
    .appName("scan-data-final-score") \
    .master("local[16]") \
    .config("spark.driver.memory", "16g") \
    .config("spark.executor.memory", "16g") \
    .config("spark.sql.execution.arrow.pyspark.enabled", "false") \
    .getOrCreate()

spark.sparkContext.setLogLevel("WARN")

input_path = "cctube_28_stage3.parquet"
output_path = "cctube_28_stage4.parquet"

df = spark.read.parquet(input_path)

# =====================
# Final score
# =====================
df = df.withColumn(
    "final_score",
    when(
        col("score") == 0,
        lit(0)
    )
    .when(
        (col("text_dominance") == "weak") &
        (col("readability_safety_clarity") == 1),
        lit(1)
    )
    .when(
        (col("text_dominance") == "strong") &
        (col("readability_safety_clarity") == 2),
        lit(2)
    )
    .otherwise(
        lit(0)
    )
)

# =====================
# Sanity check
# =====================
df.groupBy("final_score") \
  .agg(spark_count(lit(1)).alias("count")) \
  .show()

df.filter(col("final_score") == 2) \
  .select(
      "content",
      "text_dominance",
      "readability_safety_clarity",
      "safety_flag",
      "topic_clarity",
      "unreadability",
      "final_score"
  ) \
  .limit(2) \
  .show(truncate=200)

df.filter(col("final_score") == 1) \
  .orderBy(rand()) \
  .select(
      "content",
      "text_dominance",
      "readability_safety_clarity",
      "safety_flag",
      "topic_clarity",
      "unreadability",
      "final_score"
  ) \
  .limit(2) \
  .show(truncate=200)


df.filter(col("final_score") == 0) \
  .select(
      "content",
      "text_dominance",
      "readability_safety_clarity",
      "safety_flag",
      "topic_clarity",
      "unreadability",
      "final_score"
  ) \
  .limit(2) \
  .show(truncate=200)
df.write.mode("overwrite").parquet(output_path)

spark.stop()
