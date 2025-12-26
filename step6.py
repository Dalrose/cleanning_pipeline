import time
from pyspark.sql import SparkSession
from pyspark.sql.functions import (
    col, trim, regexp_replace, row_number,
    rand, lit, count as spark_count, udf, split
)
from pyspark.sql.window import Window
from pyspark.sql.types import LongType
import hashlib

# =====================
# Spark
# =====================
spark = SparkSession.builder \
    .appName("external-dedup-simhash") \
    .master("local[16]") \
    .config("spark.driver.memory", "16g") \
    .config("spark.executor.memory", "16g") \
    .config("spark.sql.execution.arrow.pyspark.enabled", "false") \
    .getOrCreate()

spark.sparkContext.setLogLevel("WARN")

input_path = "cctube_28_stage5.parquet"
output_path = "cctube_28_stage6.parquet"

df = spark.read.parquet(input_path)

# =====================
# 只对 final_score 1/2 去重
# =====================
df_valid = df.filter(col("final_score").isin([1, 2]))
df_zero = df.filter(col("final_score") == 0)

# =====================
# 文本归一化（多语言安全）
# =====================
df_norm = df_valid.withColumn(
    "norm_text",
    trim(regexp_replace(col("content"), r"\s+", " "))
)

# =====================
# Part 1️⃣ Exact duplicate（100% 相同）
# =====================
w_exact = Window.partitionBy("norm_text").orderBy(col("video_id"))

df_exact_ranked = df_norm.withColumn(
    "rn_exact", row_number().over(w_exact)
)

df_exact_kept = df_exact_ranked.filter(col("rn_exact") == 1)
df_exact_removed = df_exact_ranked.filter(col("rn_exact") > 1)

# =====================
# Part 2️⃣ SimHash（95%+ near duplicate）
# =====================

def simhash(text, bits=64):
    if not text:
        return 0
    tokens = text.split()
    v = [0] * bits
    for t in tokens:
        h = int(hashlib.md5(t.encode("utf8")).hexdigest(), 16)
        for i in range(bits):
            v[i] += 1 if (h >> i) & 1 else -1
    out = 0
    for i in range(bits):
        if v[i] > 0:
            out |= 1 << i
    return out

simhash_udf = udf(simhash, LongType())

df_sim = (
    df_exact_kept
    .withColumn("simhash", simhash_udf(col("norm_text")))
    .withColumn("bucket", col("simhash") % 1024)  # 限制 join 空间
)

# =====================
# 近似对生成（同 bucket）
# =====================
a = df_sim.alias("a")
b = df_sim.alias("b")

pairs = (
    a.join(
        b,
        (col("a.bucket") == col("b.bucket")) &
        (col("a.video_id") < col("b.video_id"))
    )
    .select(
        col("a.video_id").alias("keep_id"),
        col("b.video_id").alias("drop_id"),
        col("a.simhash").alias("hash_a"),
        col("b.simhash").alias("hash_b")
    )
)

def hamming(x, y):
    return bin(x ^ y).count("1")

hamming_udf = udf(hamming, LongType())

near_dups = (
    pairs
    .withColumn("hamming", hamming_udf(col("hash_a"), col("hash_b")))
    .filter(col("hamming") <= 2)   # ≈95%+
)

# =====================
# 要删除的 video_id
# =====================
drop_ids = near_dups.select(
    col("drop_id").alias("video_id")
).distinct()

# =====================
# Near-dup 保留 / 删除
# =====================
df_near_kept = (
    df_sim.join(drop_ids, on="video_id", how="left_anti")
    .drop("norm_text", "simhash", "bucket", "rn_exact")
)

df_near_removed = (
    df_sim.join(drop_ids, on="video_id", how="inner")
)

# =====================
# 合并 final_score = 0
# =====================
df_final = df_near_kept.unionByName(df_zero)

df_final.write.mode("overwrite").parquet(output_path)

# =====================
# 统计
# =====================
print("\n=== Final score stats ===\n")
(
    df_final.groupBy("final_score")
    .agg(spark_count(lit(1)).alias("count"))
    .orderBy("final_score")
    .show()
)

# =====================
# 展示一组被去掉的 near-duplicate
# =====================
print("\n=== Example removed near-duplicate texts ===\n")

example = (
    df_near_removed
    .select("simhash")
    .distinct()
    .limit(1)
    .collect()
)

if example:
    h = example[0]["simhash"]
    (
        df_sim.filter(col("simhash") == h)
        .select("video_id", "content", "final_score")
        .show(truncate=200)
    )
else:
    print("No near-duplicates removed.")


spark.stop()
