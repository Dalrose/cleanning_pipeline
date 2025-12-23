from pyspark.sql import SparkSession
import time
import torch
import numpy as np
from pyspark.sql.functions import (
    col, length, trim, size, split, regexp_replace, when, lit, expr, lower, count, udf
)
from pyspark.sql.types import *
import unicodedata
from langdetect import detect, detect_langs, DetectorFactory
import nltk
import re
from transformers import AutoTokenizer, AutoModelForCausalLM
nltk.data.path.append('/RLS002/Pretrain_NFS/nltk_data')
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
df = df.withColumn("score", lit(1))

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

# =====================
# Language status
# =====================
DetectorFactory.seed = 0  # 保证结果可复现

def language_status(text, transcription_language=None, threshold=0.2):
    if not text:
        return "clean"
    
    sentences = nltk.sent_tokenize(text)
    if not sentences:
        sentences = [text]
    
    lang_counts = {}
    for sent in sentences:
        try:
            lang = detect(sent)  # 返回 'en', 'fr', 'zh', ...
        except:
            lang = "unknown"
        lang_counts[lang] = lang_counts.get(lang, 0) + 1

    total = sum(lang_counts.values())
    main_lang = max(lang_counts, key=lang_counts.get)
    main_lang_ratio = lang_counts[main_lang] / total
    mix_ratio = 1 - main_lang_ratio

    if transcription_language is None:
        transcription_language = main_lang

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

df = df.filter(col("score") != 0)

# =====================
# Safety + Topic Clarity + Unreadability
# =====================
safety_schema = StructType(list(df.schema.fields) + [
    StructField("safety_flag", StringType(), True),
    StructField("topic_clarity", StringType(), True),
    StructField("unreadability", StringType(), True),
    StructField("readability_safety_clarity", IntegerType(), True)  # 中间列，用于计算最终 score
])
_generator = None
_tokenizer = None

def get_generator(model_path, dtype="float16", device="cuda"):
    global _generator, _tokenizer
    if _generator is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
        _tokenizer = AutoTokenizer.from_pretrained(model_path)
        _generator = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=getattr(torch, dtype)
        ).to(device)
    return _generator, _tokenizer

def generate_batch(prompts, model_path, max_length=256, batch_size=4, device="cuda"):
    model, tokenizer = get_generator(model_path, dtype="float16", device=device)
    results = []
    for i in range(0, len(prompts), batch_size):
        batch_texts = prompts[i:i+batch_size]
        inputs = tokenizer(batch_texts, return_tensors="pt", padding=True, truncation=True).to(device)
        outputs = model.generate(**inputs, max_new_tokens=max_length)
        for output in outputs:
            decoded = tokenizer.decode(output, skip_special_tokens=True)
            results.append(decoded)
    return results

def safety_topic_unreadability_batch(pdf_iter, batch_size=8):
    model_path = "/home/fangyi/.cache/huggingface/hub/models--Qwen--Qwen3-4B/snapshots/1cfa9a7208912126459214e8b04321603b3df60c"
    for pdf in pdf_iter:
        texts = pdf['content'].tolist()
        safety_flags, topic_clarities, unreadabilities = [], [], []

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
            decoded_list = generate_batch(prompts, model_path=model_path)

            for decoded in decoded_list:
                decoded = decoded.lower()
                safety_flags.append("unsafe" if "unsafe" in decoded else "safe")
                if "none" in decoded:
                    topic_clarities.append("none")
                elif "unclear" in decoded:
                    topic_clarities.append("unclear")
                else:
                    topic_clarities.append("clear")
                unreadabilities.append("unreadable" if "unreadable" in decoded else "readable")

        pdf['safety_flag'] = safety_flags
        pdf['topic_clarity'] = topic_clarities
        pdf['unreadability'] = unreadabilities
        pdf['readability_safety_clarity'] = pdf.apply(
            lambda row: 0 if (row['safety_flag']=='unsafe' 
                            or row['topic_clarity']=='none' 
                            or row['unreadability']=='unreadable')
                        else 1 if (row['safety_flag']=='safe' 
                                and row['topic_clarity']=='unclear' 
                                and row['unreadability']=='readable')
                        else 2,
            axis=1
        )
        yield pdf

df = df.mapInPandas(safety_topic_unreadability_batch, schema=safety_schema)

# =====================
# Score
# =====================
df = df.filter(col("score") != 0)
df = df.withColumn(
    "score",
    when(
        (col("text_dominance") == "none") |
        (col("language_status") == "conflicted") |
        (col("readability_safety_clarity") == 0),
        lit(0)
    )
    .when(
        (col("text_dominance") == "weak") |
        (col("language_status") == "mix") |
        (col("readability_safety_clarity") == 1),
        lit(1)
    )
    .otherwise(lit(2))
)

# 展示前 10 条样本
df_sample = df.select(
    "content",
    "score",
    "readability_safety_clarity",
    "safety_flag",
    "topic_clarity",
    "unreadability",
    "language_status",
    "text_dominance"
).limit(10)

df_sample.show(truncate=200)

df.write.mode("overwrite").parquet(output_path)

spark.stop()
end = time.time()
print(f"Total time: {end - start:.2f} seconds")
