import pandas as pd
import re

# -------------------------------
# STEP 1: Load Dataset
# -------------------------------
df = pd.read_csv("data/Tweets.csv")  # make sure dataset is inside 'data' folder

print("Dataset loaded ✅")
print(df.head())
print("\nSentiment distribution:\n", df["airline_sentiment"].value_counts())

# -------------------------------
# STEP 2: Clean Text
# -------------------------------
def clean_text(text):
    text = text.lower()
    text = re.sub(r"http\S+|www\S+|https\S+", "", text)  # remove links
    text = re.sub(r"@\w+|#\w+", "", text)  # remove mentions/hashtags
    text = re.sub(r"[^a-z\s]", "", text)  # keep only letters
    return text.strip()

df["clean_text"] = df["text"].apply(clean_text)

print("\nSample cleaned text:")
print(df[["text", "clean_text"]].head())

# -------------------------------
# STEP 3: Hugging Face Transformer
# -------------------------------
print("\nLoading Hugging Face sentiment model... (first time takes time)")

from transformers import pipeline

sentiment_pipeline = pipeline("sentiment-analysis")

# Try a few samples
print("\nSample Predictions (Hugging Face LLM):")
for txt in df["clean_text"].head(5).tolist():
    pred = sentiment_pipeline(txt[:200])[0]
    print(f"Tweet: {txt}\nPrediction: {pred}\n")

# -------------------------------
# STEP 4: Traditional ML Pipeline
# -------------------------------
print("\n=== Training Classical ML Model (Logistic Regression) ===")

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report

# Use cleaned text + labels
X = df["clean_text"]
y = df["airline_sentiment"]

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# TF-IDF vectorizer
vectorizer = TfidfVectorizer(max_features=5000, ngram_range=(1, 2))
X_train_tfidf = vectorizer.fit_transform(X_train)
X_test_tfidf = vectorizer.transform(X_test)

# Logistic Regression
ml_model = LogisticRegression(max_iter=200, class_weight="balanced")
ml_model.fit(X_train_tfidf, y_train)

# Predictions + evaluation
y_pred = ml_model.predict(X_test_tfidf)

print("\nML Model Accuracy:", accuracy_score(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))

# Custom predictions
sample_texts = [
    "I love flying with Delta, they are the best!",
    "My flight was delayed again, terrible service.",
    "The experience was okay, nothing special."
]
sample_tfidf = vectorizer.transform(sample_texts)
sample_preds = ml_model.predict(sample_tfidf)

print("\nSample Predictions (Classical ML):")
for txt, pred in zip(sample_texts, sample_preds):
    print(f"Text: {txt} --> Predicted Sentiment: {pred}")