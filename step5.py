from pyspark.sql.functions import col, lit, count as spark_count, rand, length, size, split, when
from pyspark.sql import SparkSession

spark = SparkSession.builder \
    .appName("external-dedup") \
    .master("local[16]") \
    .config("spark.driver.memory", "16g") \
    .config("spark.executor.memory", "16g") \
    .config("spark.sql.execution.arrow.pyspark.enabled", "false") \
    .getOrCreate()

spark.sparkContext.setLogLevel("WARN")

input_path = "cctube_28_stage4.parquet"
output_path = "cctube_28_stage5.parquet"

df = spark.read.parquet(input_path)

# =====================
# 短文本过滤
# =====================
df = df.withColumn("token_count", size(split(col("content"), r"\s+")))

df = df.withColumn(
    "final_score",
    when((col("token_count") < 8) | (length(col("content")) < 10), lit(0))
    .otherwise(col("final_score"))
)

# =====================
# 分数统计
# =====================
score_stats = (
    df.groupBy("final_score")
      .agg(spark_count(lit(1)).alias("count"))
      .orderBy("final_score")
)
score_stats.show()

# =====================
# 随机展示每个分数的前两个示例
# =====================
def show_samples(score, n=2):
    print(f"\n=== Sample texts with final_score = {score} ===\n")
    (
        df.filter(col("final_score") == score)
          .orderBy(rand())
          .select(
              "content",
              "final_score",
          )
          .limit(n)
          .show(truncate=200)
    )

show_samples(0, 2)
show_samples(1, 2)
show_samples(2, 2)

df.write.mode("overwrite").parquet(output_path)
