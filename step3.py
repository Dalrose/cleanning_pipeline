import requests
import pandas as pd
from pyspark.sql import SparkSession
from pyspark.sql.functions import col, lit, when, count as spark_count
from pyspark.sql.types import (
    StructType, StructField,
    StringType, IntegerType
)

spark = SparkSession.builder \
    .appName("scan-data-fast") \
    .master("local[16]") \
    .config("spark.driver.memory", "16g") \
    .config("spark.executor.memory", "16g") \
    .config("spark.sql.execution.arrow.pyspark.enabled", "false") \
    .getOrCreate()

spark.sparkContext.setLogLevel("WARN")

input_path = "cctube_28_stage2.parquet"
output_path = "cctube_28_stage3.parquet"

df = spark.read.parquet(input_path)

# =====================
# Safety + Topic + Readability
# =====================
safety_schema = StructType(list(df.schema.fields) + [
    StructField("safety_flag", StringType(), True),
    StructField("topic_clarity", StringType(), True),
    StructField("unreadability", StringType(), True),
    StructField("readability_safety_clarity", IntegerType(), True)
])

SERVER_URL = "http://127.0.0.1:30000/v1/chat/completions"
def generate_batch(prompts, model_path, max_length=256, batch_size=4):
    results = []

    for i in range(0, len(prompts), batch_size):
        batch = prompts[i:i+batch_size]

        payload = {
            "model": "default",
            "messages": [{"role": "user", "content": text} for text in batch],
            "max_tokens": max_length
        }
        response = requests.post(SERVER_URL, json=payload)

        if response.status_code == 200:
            decoded_list = response.json().get("choices", [])
            for decoded in decoded_list:
                results.append(decoded['message']['content'])
        else:
            results.append("Error: Unable to generate text")

    return results

def safety_topic_unreadability_batch(pdf_iter, batch_size=8):
    model_path = "/zju_0038/pengxiang/fangyi/.cache/huggingface/hub/models--Qwen--Qwen2-1.5B/snapshots/8a16abf2848eda07cc5253dec660bf1ce007ad7a"

    for pdf in pdf_iter:
        texts = pdf["content"].tolist()
        safety_flags, topic_clarities, unreadabilities = [], [], []

        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i:i+batch_size]

            prompts = [f"""
Classify the following text into three categories: Safety, Topic Clarity, and Readability.
The text may be in any language. Do NOT translate it.

Safety: SAFE or UNSAFE
Topic clarity: clear / unclear / none
Readability: readable / unreadable

Text:
{text}

Answer format:
SAFETY, TOPIC, READABILITY
""" for text in batch_texts]

            decoded_list = generate_batch(prompts, model_path)

            for decoded in decoded_list:
                d = decoded.lower()
                safety_flags.append("unsafe" if "unsafe" in d else "safe")

                if "none" in d:
                    topic_clarities.append("none")
                elif "unclear" in d:
                    topic_clarities.append("unclear")
                else:
                    topic_clarities.append("clear")

                unreadabilities.append(
                    "unreadable" if "unreadable" in d else "readable"
                )

        if len(safety_flags) < len(pdf):
            safety_flags.extend([None] * (len(pdf) - len(safety_flags)))
        if len(topic_clarities) < len(pdf):
            topic_clarities.extend([None] * (len(pdf) - len(topic_clarities)))
        if len(unreadabilities) < len(pdf):
            unreadabilities.extend([None] * (len(pdf) - len(unreadabilities)))

        pdf["safety_flag"] = safety_flags
        pdf["topic_clarity"] = topic_clarities
        pdf["unreadability"] = unreadabilities

        pdf["readability_safety_clarity"] = pdf.apply(
            lambda r: 0 if (
                r["safety_flag"] == "unsafe"
                or r["topic_clarity"] == "none"
                or r["unreadability"] == "unreadable"
            )
            else 1 if (
                r["safety_flag"] == "safe"
                and r["topic_clarity"] == "unclear"
                and r["unreadability"] == "readable"
            )
            else 2,
            axis=1
        )

        yield pdf


df = df.mapInPandas(
    safety_topic_unreadability_batch,
    schema=safety_schema
)

# =====================
# score
# =====================
df.groupBy("readability_safety_clarity") \
  .agg(spark_count(lit(1)).alias("count")) \
  .show()

df.filter(col("readability_safety_clarity") == 0) \
  .select(
      "content",
      "safety_flag",
      "topic_clarity",
      "unreadability",
      "readability_safety_clarity"
  ) \
  .limit(5) \
  .show(truncate=200)

df.write.mode("overwrite").parquet(output_path)
