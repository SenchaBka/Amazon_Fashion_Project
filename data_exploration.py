import os, json, gzip, re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Load data
DATA_PATH = "data/AMAZON_FASHION.json" 

def load_jsonl(path: str, nrows: int | None = None) -> pd.DataFrame:
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            if nrows is not None and i >= nrows:
                break
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return pd.DataFrame(rows)

df = load_jsonl(DATA_PATH, nrows=None)  

print("Loaded:", df.shape)
print("Columns:", df.columns.tolist())

# Combine text: reviewText + summary if available
if "reviewText" not in df.columns:
    df["reviewText"] = ""
if "summary" in df.columns:
    df["text_full"] = (df["summary"].fillna("") + " " + df["reviewText"].fillna("")).str.strip()
else:
    df["text_full"] = df["reviewText"].fillna("").astype(str)

# Votes cleaning and parsing
df["votes"] = pd.to_numeric(
    df["vote"].astype(str).str.replace(",", ""), 
    errors="coerce"
)

# Time parsing
df["date"] = pd.to_datetime(df["unixReviewTime"], unit="s")

# Basic counts & averages
print("\n--- Basic Exploration ---")
print("Total reviews:", len(df))
print("Unique products (asin):", df["asin"].nunique())
print("Unique users:", df["reviewerID"].nunique())
print("Average rating:", df["overall"].mean())
print("Median rating:", df["overall"].median())
print(df["overall"].value_counts(normalize=True))

reviews_per_product = df.groupby("asin").size()
reviews_per_user = df.groupby("reviewerID").size()

print("Avg reviews per product:", reviews_per_product.mean())
print("Median reviews per product:", reviews_per_product.median())
print("Avg reviews per user:", reviews_per_user.mean())
print("Median reviews per user:", reviews_per_user.median())

# Rating distribution
plt.figure()
df["overall"].value_counts().sort_index().plot(kind="bar")
plt.title("Rating distribution")
plt.xlabel("Rating")
plt.ylabel("Count")
plt.tight_layout()
plt.show()

# Distribution: # reviews per product (long tail)
plt.figure()
reviews_per_product.plot(kind="hist", bins=50, log=True)
plt.title("Distribution of reviews per product (log y)")
plt.xlabel("Reviews per product")
plt.ylabel("Frequency (log)")
plt.tight_layout()
plt.show()

plt.figure()
reviews_per_product.sort_values(ascending=False).reset_index(drop=True).plot()
plt.title("Reviews per product (ranked) — long tail")
plt.xlabel("Product rank")
plt.ylabel("Reviews")
plt.tight_layout()
plt.show()

# Distribution: # reviews per user
plt.figure()
reviews_per_user.plot(kind="hist", bins=50, log=True)
plt.title("Distribution of reviews per user (log y)")
plt.xlabel("Reviews per user")
plt.ylabel("Frequency (log)")
plt.tight_layout()
plt.show()

plt.figure()
reviews_per_user.sort_values(ascending=False).reset_index(drop=True).plot()
plt.title("Reviews per user (ranked) — activity long tail")
plt.xlabel("User rank")
plt.ylabel("Reviews")
plt.tight_layout()
plt.show()

# Review length + outliers
def word_count(s: str) -> int:
    if not isinstance(s, str):
        return 0
    # split on whitespace
    return len(s.split())

df["char_len"] = df["text_full"].fillna("").astype(str).str.len()
df["word_len"] = df["text_full"].fillna("").astype(str).apply(word_count)

print("\n--- Review length stats ---")
print(df[["char_len", "word_len"]].describe(percentiles=[0.5, 0.75, 0.9, 0.95, 0.99]))

plt.figure()
df["word_len"].plot(kind="hist", bins=60, log=True)
plt.title("Review word length distribution (log y)")
plt.xlabel("Words per review")
plt.ylabel("Frequency (log)")
plt.tight_layout()
plt.show()

plt.figure()
df["char_len"].plot(kind="hist", bins=60, log=True)
plt.title("Review character length distribution (log y)")
plt.xlabel("Characters per review")
plt.ylabel("Frequency (log)")
plt.tight_layout()
plt.show()

# Outliers using IQR
q1, q3 = df["word_len"].quantile([0.25, 0.75])
iqr = q3 - q1
upper = q3 + 1.5 * iqr
lower = max(0, q1 - 1.5 * iqr)

outliers = df[(df["word_len"] > upper) | (df["word_len"] < lower)]
print("\nOutlier thresholds (words):", {"lower": float(lower), "upper": float(upper)})
print("Outlier review count:", len(outliers))

print("\nSample longest reviews:")
print(df.sort_values("word_len", ascending=False)[["asin", "reviewerID", "overall", "word_len", "text_full"]].head(3).to_string(index=False))


# ----------------------------
# 7) Duplicates checks
# ----------------------------
print("\n--- Duplicate checks ---")

# 7a) Exact duplicate rows
dup_pairs = df.duplicated(subset=["reviewerID", "asin", "unixReviewTime"])
print("Duplicate (reviewerID, asin, unixReviewTime):", dup_pairs)

# 7b) Same user + same product duplicates
if "date" in df.columns:
    dup_user_product = df.duplicated(subset=["reviewerID", "asin"]).sum()
else:
    dup_user_product = df.duplicated(subset=["reviewerID", "asin"]).sum()
print("Duplicate (reviewerID, asin) pairs:", dup_user_product)

# 7c) Identical text duplicates (possible copied reviews)
df["text_norm"] = (
    df["text_full"]
    .fillna("")
    .astype(str)
    .str.lower()
    .str.replace(r"\s+", " ", regex=True)
    .str.strip()
)
dup_text = df.duplicated(subset=["text_norm"]).sum()
print("Duplicate normalized text:", dup_text)

# show top repeated texts (excluding empty)
text_counts = df[df["text_norm"] != ""].groupby("text_norm").size().sort_values(ascending=False)
print("\nTop repeated review texts:")
print(text_counts.head(10))


# ----------------------------
# 8) Verified vs non-verified (if available)
# ----------------------------
if df["verified"].notna().any():
    tmp = df.dropna(subset=["verified"])
    grp = tmp.groupby("verified")["overall"].agg(["count", "mean", "median"])
    print("\n--- Verified vs non-verified ratings ---")
    print(grp)

    plt.figure()
    grp["mean"].plot(kind="bar")
    plt.title("Average rating: verified vs non-verified")
    plt.xlabel("Verified purchase")
    plt.ylabel("Average rating")
    plt.tight_layout()
    plt.show()
else:
    print("\nNo verified column (or all missing). Skipping verified analysis.")


# ----------------------------
# 9) Helpful votes analysis (if available)
# ----------------------------
if df["helpful_votes"].notna().any():
    hv = df["helpful_votes"].dropna()
    print("\n--- Helpful votes stats ---")
    print(hv.describe(percentiles=[0.5, 0.9, 0.95, 0.99]))

    plt.figure()
    hv.plot(kind="hist", bins=60, log=True)
    plt.title("Helpful votes distribution (log y)")
    plt.xlabel("Helpful votes")
    plt.ylabel("Frequency (log)")
    plt.tight_layout()
    plt.show()

    # Correlation between length and helpful votes
    merged = df.dropna(subset=["helpful_votes"])
    if len(merged) > 100:
        corr = merged[["helpful_votes", "word_len", "overall"]].corr(numeric_only=True)
        print("\nCorrelation matrix (helpful_votes, word_len, rating):")
        print(corr)
else:
    print("\nNo helpful votes data. Skipping helpfulness analysis.")


# ----------------------------
# 10) Time trends (if date available)
# ----------------------------
if df["date"].notna().any():
    df_time = df.dropna(subset=["date"]).copy()
    df_time["month"] = df_time["date"].dt.to_period("M").dt.to_timestamp()

    reviews_by_month = df_time.groupby("month").size()

    plt.figure()
    reviews_by_month.plot()
    plt.title("Reviews over time (monthly)")
    plt.xlabel("Month")
    plt.ylabel("Number of reviews")
    plt.tight_layout()
    plt.show()

    # seasonal signal (month-of-year)
    df_time["month_of_year"] = df_time["date"].dt.month
    moy = df_time.groupby("month_of_year").size()

    plt.figure()
    moy.plot(kind="bar")
    plt.title("Seasonality: reviews by month-of-year")
    plt.xlabel("Month")
    plt.ylabel("Number of reviews")
    plt.tight_layout()
    plt.show()
else:
    print("\nNo date field available. Skipping temporal analysis.")


# ----------------------------
# 11) Creative: length vs rating (does negativity produce longer reviews?)
# ----------------------------
plt.figure()
df.boxplot(column="word_len", by="overall")
plt.title("Review length (words) by rating")
plt.suptitle("")
plt.xlabel("Rating")
plt.ylabel("Words")
plt.tight_layout()
plt.show()

# Optional: show mean/median length by rating
len_by_rating = df.groupby("overall")["word_len"].agg(["count", "mean", "median"]).sort_index()
print("\n--- Review length by rating ---")
print(len_by_rating)


# ----------------------------
# 12) Creative: Cold start severity
# ----------------------------
cold_products = (reviews_per_product <= 2).mean()
cold_users = (reviews_per_user <= 1).mean()
print("\n--- Cold-start indicators ---")
print(f"Share of products with <=2 reviews: {cold_products:.2%}")
print(f"Share of users with 1 review: {cold_users:.2%}")

# Verified vs Non-Verified Behavior
df.groupby("verified")["overall"].agg(["count", "mean"])

# Text Complexity (not just length)
df["avg_word_len"] = df["text_full"].apply(lambda x: np.mean([len(w) for w in str(x).split()]) if x else 0)

df["avg_word_len"].hist(bins=50)
plt.title("Average word length distribution")
plt.show()

# Repetitions
df["text_norm"] = df["text_full"].str.lower().str.strip()

duplicate_texts = df["text_norm"].value_counts().head(10)
print(duplicate_texts)


# Reviews Over Time
df["date"] = pd.to_datetime(df["unixReviewTime"], unit="s")

df.groupby(df["date"].dt.year).size().plot()
plt.title("Reviews over time")
plt.show()

# Helpful Votes vs Review Length
df.dropna(subset=["helpful_votes"]).plot.scatter(x="word_len", y="helpful_votes")
plt.title("Helpful votes vs review length")
plt.show()

# Bias
df["overall"].value_counts(normalize=True)

# User behaviour Segmentation
user_counts = df.groupby("reviewerID").size()

print("1 review users:", (user_counts == 1).mean())
print("Power users (>10):", (user_counts > 10).mean())