
import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report

# Load dataset
data = pd.read_csv("news.csv")

# Check required columns
required = {"text", "label"}
if not required.issubset(data.columns):
    raise ValueError("CSV must contain 'text' and 'label' columns.")

# Clean dataset
data = data.dropna(subset=["text", "label"])
data["text"] = data["text"].astype(str).str.strip()
data["label"] = data["label"].astype(str).str.strip().str.upper()
data = data[data["label"].isin(["FAKE", "REAL"])]
data = data[data["text"] != ""]

if data["label"].nunique() != 2:
    raise ValueError("Dataset must contain both FAKE and REAL labels.")

# Split training and testing data
X_train, X_test, y_train, y_test = train_test_split(
    data["text"],
    data["label"],
    test_size=0.2,
    random_state=42,
    stratify=data["label"]
)

# Build the machine learning pipeline
model = Pipeline([
    ("tfidf", TfidfVectorizer(
        stop_words="english",
        max_df=0.95,
        min_df=2,
        ngram_range=(1, 2),
        max_features=50000
    )),
    ("classifier", LogisticRegression(max_iter=1000))
])

# Train the model
model.fit(X_train, y_train)

# Evaluate on unseen test examples
predictions = model.predict(X_test)

print("Accuracy:", round(accuracy_score(y_test, predictions) * 100, 2), "%")
print("\nClassification Report:")
print(classification_report(y_test, predictions, zero_division=0))

# Save the trained model
joblib.dump(model, "fake_news_model.joblib")

print("\nModel saved as fake_news_model.joblib")
