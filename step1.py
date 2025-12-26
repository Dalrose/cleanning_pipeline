from pyspark.sql import SparkSession
import time
from pyspark.sql.functions import col, trim, lit, when, udf
from pyspark.sql.types import IntegerType
from pyspark.sql.functions import count as spark_count
import re

spark = SparkSession.builder \
    .appName("scan-data-fast") \
    .master("local[16]") \
    .config("spark.driver.memory", "16g") \
    .config("spark.executor.memory", "16g") \
    .config("spark.sql.execution.arrow.pyspark.enabled", "false") \
    .getOrCreate()

spark.sparkContext.setLogLevel("WARN")

input_path = "cctube_28.parquet"
output_path = "cctube_28_stage1.parquet"

df = spark.read.parquet(input_path).withColumn("content", trim(col("text")))
df = df.withColumn("score", lit(1))

# =====================
# Content status: 文本内部去重
# =====================
def count_duplicate_sentences(text):
    if not text:
        return 0
    sentences = re.split(r'(?<=[.!?])\s+', text)
    seen = set()
    dup = 0
    for s in sentences:
        if s in seen:
            dup += 1
        else:
            seen.add(s)
    return dup
dup_count_udf = udf(count_duplicate_sentences, IntegerType())

df = df.withColumn("duplicate_count", dup_count_udf(col("content")))
df = df.withColumn(
    "score",
    when(col("duplicate_count") > 5, lit(0))
    .otherwise(col("score"))
)

score_stats = df.groupBy("score").agg(spark_count(lit(1)).alias("count"))
score_stats.show()

df_zero_sample = df.filter(col("score") == 0).select("content", "duplicate_count", "score").limit(5)
df_zero_sample.show(truncate=200)

df.write.mode("overwrite").parquet(output_path)
