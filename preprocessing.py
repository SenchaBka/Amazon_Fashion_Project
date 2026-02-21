import json
import re
import numpy as np
import pandas as pd

DATA_PATH = "data/AMAZON_FASHION.json" 

# ----------------------------
# 0) Load JSONL (one JSON object per line)
# ----------------------------
df = pd.read_json(DATA_PATH, lines=True)
print("Loaded:", df.shape)
print("Columns:", df.columns.tolist())

# ----------------------------
# 1) (a) Label sentiment from rating ("overall")
# ----------------------------
# Ratings 4,5 -> Positive
# Rating 3   -> Neutral
# Ratings 1,2 -> Negative
def label_sentiment(r):
    try:
        r = float(r)
    except Exception:
        return np.nan
    if r >= 4:
        return "Positive"
    elif r == 3:
        return "Neutral"
    elif r <= 2:
        return "Negative"
    return np.nan

if "overall" not in df.columns:
    raise KeyError("Expected column 'overall' (rating) not found.")

df["sentiment"] = df["overall"].apply(label_sentiment)

# Drop rows with invalid rating/label
df = df.dropna(subset=["sentiment"]).copy()

# ----------------------------
# 2) (b) Choose columns for sentiment analyzer
# ----------------------------
# We use:
# - reviewText: main opinion content (highest signal for sentiment)
# - summary: short title often contains strong sentiment words
# We combine them into a single feature `text` used by the sentiment model.

# Ensure columns exist
if "reviewText" not in df.columns:
    df["reviewText"] = ""
if "summary" not in df.columns:
    df["summary"] = ""

df["text_raw"] = (df["summary"].fillna("").astype(str) + " " + df["reviewText"].fillna("").astype(str)).str.strip()

# ----------------------------
# 3) Minimal text cleaning (basic preprocessing)
# ----------------------------
# Keep it simple and safe:
# - lowercasing
# - remove URLs
# - remove extra whitespace
# - keep punctuation (often useful for sentiment), but remove non-printing chars

url_re = re.compile(r"http\S+|www\.\S+")
ws_re = re.compile(r"\s+")

def clean_text(s: str) -> str:
    s = str(s)
    s = s.replace("\u200b", " ").replace("\ufeff", " ")   # zero-width / BOM
    s = url_re.sub("", s)
    s = s.lower()
    s = ws_re.sub(" ", s).strip()
    return s

df["text"] = df["text_raw"].apply(clean_text)

# Optional: drop empty text rows (no info for sentiment)
df = df[df["text"].str.len() > 0].copy()

# ----------------------------
# 4) (c) Outlier checks
# ----------------------------
# Primary outliers for NLP are usually review lengths:
# - extremely short reviews (low info)
# - extremely long reviews (possible spam/noise)
#
# We compute:
# - word_len and char_len
# - IQR outliers
# - top extremes

df["word_len"] = df["text"].apply(lambda x: len(str(x).split()))
df["char_len"] = df["text"].apply(lambda x: len(str(x)))

# IQR method on word length
q1 = df["word_len"].quantile(0.25)
q3 = df["word_len"].quantile(0.75)
iqr = q3 - q1
lower = max(0, q1 - 1.5 * iqr)
upper = q3 + 1.5 * iqr

df["is_len_outlier"] = (df["word_len"] < lower) | (df["word_len"] > upper)

# Additional practical flags (optional but useful)
df["is_too_short"] = df["word_len"] <= 2
df["is_too_long"] = df["word_len"] >= 250

# Summary stats
print("\n=== Sentiment label distribution ===")
print(df["sentiment"].value_counts())

print("\n=== Text length stats ===")
print(df[["word_len", "char_len"]].describe(percentiles=[0.5, 0.75, 0.9, 0.95, 0.99]))

print("\n=== IQR thresholds (word_len) ===")
print({"lower": float(lower), "upper": float(upper)})

print("\nOutlier counts:")
print("IQR length outliers:", int(df["is_len_outlier"].sum()))
print("Too short (<=2 words):", int(df["is_too_short"].sum()))
print("Too long (>=250 words):", int(df["is_too_long"].sum()))

# Show examples of extremes
print("\n--- Examples: shortest reviews ---")
print(df.sort_values("word_len", ascending=True)[["overall", "sentiment", "word_len", "text"]].head(10).to_string(index=False))

print("\n--- Examples: longest reviews ---")
print(df.sort_values("word_len", ascending=False)[["overall", "sentiment", "word_len", "text"]].head(5).to_string(index=False))

# ----------------------------
# 5) Output a clean modeling table (recommended)
# ----------------------------
# Keep only what your sentiment model needs:
# - text (features)
# - sentiment (label)
# Optionally keep metadata (asin, verified, time) for later analysis

cols_to_keep = ["text", "sentiment", "overall"]
for c in ["asin", "verified", "unixReviewTime", "reviewTime", "reviewerID"]:
    if c in df.columns:
        cols_to_keep.append(c)

model_df = df[cols_to_keep].copy()

print("\nFinal modeling dataframe shape:", model_df.shape)
print(model_df.head(3))

# Save for next steps (train/test, vectorization, etc.)
model_df.to_csv("amazon_fashion_preprocessed.csv", index=False)
print("\nSaved: amazon_fashion_preprocessed.csv")