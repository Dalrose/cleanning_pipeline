from pyspark.sql import SparkSession
import time
import torch
import numpy as np
from pyspark.sql.functions import (
    col, length, trim, size, split, regexp_replace, when, lit, expr, lower, count, udf
)
from pyspark.sql.types import StringType, IntegerType, ArrayType, StructType, StructField
from transformers import AutoTokenizer, AutoModelForCausalLM
import unicodedata
import fasttext
import nltk
import re

start = time.time()

spark = SparkSession.builder \
    .appName("scan-data-fast") \
    .master("local[16]") \
    .config("spark.sql.execution.arrow.pyspark.enabled", "false") \
    .getOrCreate()
spark.sparkContext.setLogLevel("WARN")

input_path = "/RLS002/Pretrain_NFS/dataset/PleIAs/YouTube-Commons/cctube_28.parquet"
output_path = "/RLS002/Pretrain_NFS/dataset/PleIAs/YouTube-Commons-Cleaned/cctube_28.parquet"

df = spark.read.parquet(input_path).withColumn("content", trim(col("text")))

# =====================
# Content status: 文本内部去重
# =====================
def count_duplicate_sentences(text):
    if not text:
        return 0
    sentences = nltk.sent_tokenize(text)
    seen = set()
    duplicates = 0
    for s in sentences:
        if s in seen:
            duplicates += 1
        else:
            seen.add(s)
    return duplicates

dup_count_udf = udf(count_duplicate_sentences, IntegerType())

df = df.withColumn("duplicate_count", dup_count_udf(col("content")))
df = df.withColumn(
    "score",
    when(col("duplicate_count") > 5, lit(0))
    .otherwise(col("score"))
)

# =====================
# Text dominance
# =====================
df = df.withColumn("text_len", length("content"))
df = df.withColumn("word_count", size(split("content", r"\s+")))

pattern = r"\[.*?\]"

def extract_all_matches(text):
    if not text:
        return []
    return re.findall(pattern, text)

regexp_extract_all_udf = udf(extract_all_matches, ArrayType(StringType()))

df = df.withColumn("music_matches", regexp_extract_all_udf(col("content")))
df = df.withColumn("music_count", size(col("music_matches")))

df = df.withColumn(
    "music_ratio",
    col("music_count") / (col("word_count") + lit(1))
)

df = df.withColumn(
    "text_dominance",
    when((col("text_len") < 10) | (col("music_ratio") >= 0.2), "none")
    .when((col("music_ratio") >= 0.1) & (col("music_ratio") < 0.2), "weak")
    .otherwise("strong")
)

df = df.withColumn(
    "score",
    when(col("text_dominance") == "none", lit(0))
    .when(col("text_dominance") == "weak", lit(1))
    .otherwise(col("score"))
)

df = df.filter(col("score") != 0)

# =====================
# Language status
# =====================
model = fasttext.load_model("lid.176.bin")
def language_status(text, transcription_language=None, threshold=0.2):
    if not text:
        return "clean"
    
    sentences = nltk.sent_tokenize(text)
    if not sentences:
        sentences = [text]
    
    lang_counts = {}
    for sent in sentences:
        lang = model.predict(sent)[0][0]
        lang_counts[lang] = lang_counts.get(lang, 0) + 1
    
    total = sum(lang_counts.values())
    main_lang = max(lang_counts, key=lang_counts.get)
    main_lang_ratio = lang_counts[main_lang] / total
    mix_ratio = 1 - main_lang_ratio

    if mix_ratio >= threshold:
        return "mix"
    elif transcription_language != main_lang:
        return "conflicted"
    else:
        return "clean"

language_status_udf = udf(language_status, StringType())

df = df.withColumn(
    "language_status",
    language_status_udf(col("content"), col("transcription_language"))
)

df = df.withColumn(
    "score",
    when(col("language_status") == "conflicted", lit(0)).otherwise(col("score"))
)

# =====================
# Safety + Topic Clarity + Unreadability
# =====================
model_name = "meta-llama/Meta-Llama-Guard-2-8B"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype=torch.float16,
    device_map="auto"
)


def safety_topic_unreadability_batch(pdf_iter, batch_size=16):
    results = []
    for pdf in pdf_iter:
        texts = pdf['content'].tolist()
        safety_flags = []
        topic_clarities = []
        unreadabilities = []

        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i:i+batch_size]

            prompts = [f"""
Classify the following text into three categories: Safety, Topic Clarity, and Readability.
The text may be in any language. Do NOT translate it.

1. Safety risk:
   - SAFE → Text is non-harmful, does not contain violence, hate speech, illegal content, or sensitive information.
   - UNSAFE → Text may contain harmful, illegal, or sensitive content.

2. Topic clarity:
   - clear → Content is complete, sentences are well-formed, topic is coherent and understandable.
   - unclear → Content is incomplete, vague, or partially understandable.
   - none → No meaningful content, purely emotional, repetitive, or nonsensical.

3. Readability / Unreadability:
   - readable → Sentences can be read and understood; logical flow exists.
   - unreadable → Text is garbled, incoherent, or extremely fragmented.

Text:
{text}

Answer format (comma-separated, only one word each):
SAFETY, TOPIC, READABILITY
""" for text in batch_texts]

            inputs = tokenizer(prompts, return_tensors="pt", padding=True, truncation=True).to(model.device)
            outputs = model.generate(**inputs, max_new_tokens=64, do_sample=False)

            decoded_list = [tokenizer.decode(o, skip_special_tokens=True).lower() for o in outputs]

            for decoded in decoded_list:
                # Safety
                safety_flag = "unsafe" if "unsafe" in decoded else "safe"

                # Topic clarity
                if "none" in decoded:
                    topic_clarity = "none"
                elif "unclear" in decoded:
                    topic_clarity = "unclear"
                else:
                    topic_clarity = "clear"

                # Readability
                unreadability = "unreadable" if "unreadable" in decoded else "readable"

                safety_flags.append(safety_flag)
                topic_clarities.append(topic_clarity)
                unreadabilities.append(unreadability)

        pdf['safety_flag'] = safety_flags
        pdf['topic_clarity'] = topic_clarities
        pdf['unreadability'] = unreadabilities

        pdf['score'] = pdf.apply(
            lambda row: 0 if (row['safety_flag']=='unsafe' or row['topic_clarity']=='none' or row['unreadability']=='unreadable') else 1,
            axis=1
        )

        yield pdf

df = df.mapInPandas(safety_topic_unreadability_batch, schema=safety_schema.add(StructField("score", IntegerType())))

# =====================
# Score
# =====================
df = df.filter(col("score") != 0)

df = df.withColumn(
    "score",
    when(
        (col("text_dominance") == "weak") |
        (col("language_status") == "mix") |
        (col("topic_clarity") == "unclear"),
        lit(1)
    )
    .otherwise(lit(2))
)

df.write.mode("overwrite").parquet(output_path)

spark.stop()
end = time.time()
print(f"Total time: {end - start:.2f} seconds")
