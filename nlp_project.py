import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import re
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from textblob import TextBlob
from sklearn.metrics import confusion_matrix

# Load the Dataset
file_path = 'data/AMAZON_FASHION_5.json'
df = pd.read_json(file_path, lines=True)


# a. Counts, averages
print("--- Counts and Averages ---")
total_reviews = len(df)
unique_users = df['reviewerID'].nunique()
unique_products = df['asin'].nunique()
average_rating = df['overall'].mean()

print(f"Total number of reviews: {total_reviews:,}")
print(f"Total unique users: {unique_users:,}")
print(f"Total unique products: {unique_products:,}")
print(f"Average product rating: {average_rating:.2f} / 5.0")


# b, c. Distribution of the number of reviews across/per products
print("--- Reviews per Product ---")
reviews_per_product = df['asin'].value_counts()

print("Basic statistics for reviews per product:")
print(reviews_per_product.describe())


# d. Distribution of reviews per user
print("--- Reviews per User ---")
reviews_per_user = df['reviewerID'].value_counts()



print("Basic statistics for reviews per user:")
print(reviews_per_user.describe())


# e,f. Review lengths, analyze lengths, and outliers
print("--- Review Lengths and Outliers ---")
# Calculate the length of each review by word count
df['review_word_count'] = df['reviewText'].apply(lambda x: len(str(x).split()))

print("Basic statistics for review word counts:")
print(df['review_word_count'].describe())

# 1. Histogram to analyze the distribution of review lengths
plt.figure(figsize=(10, 5))
# Clipping at the 99th percentile for a cleaner histogram, as extreme outliers will skew the plot
upper_limit = df['review_word_count'].quantile(0.99)
sns.histplot(df['review_word_count'], bins=50, color='purple')
plt.title('Distribution of Review Word Counts')
plt.xlabel('Number of Words')
plt.ylabel('Frequency')
plt.show()


# 2. Boxplot to visualize outliers
plt.figure(figsize=(10, 3))
sns.boxplot(x=df['review_word_count'], color='orange')
plt.title('Boxplot of Review Lengths (Showing Outliers)')
plt.xlabel('Number of Words')
plt.show()


# Identify how many outliers there are (using the standard IQR method)
Q1 = df['review_word_count'].quantile(0.25)
Q3 = df['review_word_count'].quantile(0.75)
IQR = Q3 - Q1
outlier_threshold = Q3 + 1.5 * IQR
num_outliers = len(df[df['review_word_count'] > outlier_threshold])

print(f"Threshold for outliers (Q3 + 1.5*IQR): {outlier_threshold:.2f} words")
print(f"Number of reviews considered outliers (extremely long): {num_outliers:,} out of {total_reviews:,}")


# Check for duplicates
print("--- Duplicate Checks ---")

# Check for exact duplicate reviews by looking at core text and ID columns
core_columns = ['reviewerID', 'asin', 'overall', 'reviewText', 'unixReviewTime']
exact_duplicates = df.duplicated(subset=core_columns).sum()
print(f"Total exact duplicate reviews (matching user, product, and text): {exact_duplicates}")

# Check for logical duplicates: The same user reviewing the same product more than once
user_product_duplicates = df.duplicated(subset=['reviewerID', 'asin'], keep=False).sum()
print(f"Total cases where the same user reviewed the same product multiple times: {user_product_duplicates}")

# Check for duplicate review texts (excluding empty reviews)
non_empty_reviews = df[df['reviewText'].str.strip() != ''].copy()
duplicate_texts = non_empty_reviews.duplicated(subset=['reviewText']).sum()
print(f"Total duplicate review texts (same text used multiple times): {duplicate_texts}")


# Label the Data
def label_sentiment(rating):
    if rating >= 4:
        return 'Positive'
    elif rating == 3:
        return 'Neutral'
    else:
        return 'Negative'

df['sentiment_label'] = df['overall'].apply(label_sentiment)


# Selecting Columns
# We keep reviewText for analysis and overall/sentiment label for validation
df_processed = df[['reviewText', 'overall', 'sentiment_label']].copy()


# Addressing Outliers
# Removing empty reviews or those that are just whitespace
df_processed = df_processed[df_processed['reviewText'].str.strip() != '']


# Text Pre processing
def clean_text_for_lexicon(text):
    # Convert input to string to handle potential NaN/float values
    text = str(text)
    
    # Convert to lowercase
    text = text.lower()
    
    # Remove numbers
    text = re.sub(r'\d+', '', text)
    
    # Remove extra whitespace
    text = " ".join(text.split())
    
    return text

df_processed['cleaned_review'] = df_processed['reviewText'].apply(clean_text_for_lexicon)
df_processed.drop_duplicates(subset=['reviewText'], inplace=True)

print("Pre-processing Complete. Preview of labeled data:")
print(df_processed[['cleaned_review', 'sentiment_label']].head(20))
print(f"number of reviews after preprocessing {len(df_processed)}")


# Randomly select 1000 reviews

# We use a random_state to ensure the results are reproducible, we ended up with 439 instead of 1000 reviews so we had to sample 400
df_sample = df_processed.sample(n=400, random_state=98).copy()


# 6. Modeling (Sentiment Analysis) Lexicon approach
# Initialize VADER
analyzer = SentimentIntensityAnalyzer()

def get_vader_label(text):
    # compound score ranges from -1 (Extremely Negative) to 1 (Extremely Positive)
    score = analyzer.polarity_scores(text)['compound']
    if score >= 0.05:
        return 'Positive'
    elif score <= -0.05:
        return 'Negative'
    else:
        return 'Neutral'

# Initialize TextBlob logic
def get_textblob_label(text):
    # polarity ranges from -1 to 1
    polarity = TextBlob(text).sentiment.polarity
    if polarity > 0:
        return 'Positive'
    elif polarity < 0:
        return 'Negative'
    else:
        return 'Neutral'

# Apply models to the 400 review sample
df_sample['vader_pred'] = df_sample['cleaned_review'].apply(get_vader_label)
df_sample['textblob_pred'] = df_sample['cleaned_review'].apply(get_textblob_label)


# Validate results and Comparison Table

# Calculate Accuracy for both
vader_accuracy = (df_sample['vader_pred'] == df_sample['sentiment_label']).mean()
textblob_accuracy = (df_sample['textblob_pred'] == df_sample['sentiment_label']).mean()

# Create comparison table
comparison_data = {
    'Approach': ['VADER', 'TextBlob'],
    'Accuracy': [f"{vader_accuracy:.2%}", f"{textblob_accuracy:.2%}"],
    'Pre-processing Used': ['Lowercase, No Numbers', 'Lowercase, No Numbers']
}

comparison_table = pd.DataFrame(comparison_data)

print("--- Validation: Comparison Table ---")
print(comparison_table)


# Define the order of labels for the matrix to ensure consistency
labels = ['Positive', 'Neutral', 'Negative']

# Generate Confusion Matrices
cm_vader = confusion_matrix(df_sample['sentiment_label'], df_sample['vader_pred'], labels=labels)
cm_textblob = confusion_matrix(df_sample['sentiment_label'], df_sample['textblob_pred'], labels=labels)


# Plotting the Results
fig, ax = plt.subplots(1, 2, figsize=(15, 6))

# Plot VADER Confusion Matrix
sns.heatmap(cm_vader, annot=True, fmt='d', cmap='Blues', ax=ax[0],
            xticklabels=labels, yticklabels=labels)
ax[0].set_title('VADER Confusion Matrix')
ax[0].set_ylabel('Actual Label')
ax[0].set_xlabel('Predicted Label')

# Plot TextBlob Confusion Matrix
sns.heatmap(cm_textblob, annot=True, fmt='d', cmap='Greens', ax=ax[1],
            xticklabels=labels, yticklabels=labels)
ax[1].set_title('TextBlob Confusion Matrix')
ax[1].set_ylabel('Actual Label')
ax[1].set_xlabel('Predicted Label')

plt.tight_layout()
plt.show()