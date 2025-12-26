import re
from pyspark.sql import SparkSession
from pyspark.sql.functions import (
    col, lit, when, udf,
    length, size, split,
    count as spark_count
)
from pyspark.sql.types import ArrayType, StringType

spark = SparkSession.builder \
    .appName("scan-data-fast") \
    .master("local[16]") \
    .config("spark.driver.memory", "16g") \
    .config("spark.executor.memory", "16g") \
    .config("spark.sql.execution.arrow.pyspark.enabled", "false") \
    .getOrCreate()

spark.sparkContext.setLogLevel("WARN")

input_path = "cctube_28_stage1.parquet"
output_path = "cctube_28_stage2.parquet"

df = spark.read.parquet(input_path)

# =====================
# Text dominance
# =====================
df = df.withColumn("text_len", length(col("content")))
df = df.withColumn("word_count", size(split(col("content"), r"\s+")))

pattern = r"\[.*?\]"

def extract_all_matches(text):
    if not text:
        return []
    return re.findall(pattern, text)

extract_udf = udf(extract_all_matches, ArrayType(StringType()))

df = df.withColumn("music_matches", extract_udf(col("content")))
df = df.withColumn("music_count", size(col("music_matches")))

df = df.withColumn(
    "music_ratio",
    col("music_count") / (col("word_count") + lit(1))
)

df = df.withColumn(
    "text_dominance",
    when((col("text_len") < 10) | (col("music_ratio") >= 0.2), lit("none"))
    .when((col("music_ratio") >= 0.1), lit("weak"))
    .otherwise(lit("strong"))
)

df = df.withColumn(
    "score",
    when(col("text_dominance") == "none", lit(0))
    .otherwise(col("score"))
)

# =====================
# score 统计
# =====================
score_stats = df.groupBy("score").agg(spark_count(lit(1)).alias("count"))
score_stats.show()

df_zero_sample = (
    df.filter(col("score") == 0)
      .select(
          "content",
          "text_len",
          "word_count",
          "music_count",
          "music_ratio",
          "text_dominance",
          "score"
      )
      .limit(5)
)

df_zero_sample.show(truncate=200)

df.write.mode("overwrite").parquet(output_path)
