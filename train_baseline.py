import numpy as np
import pandas as pd
from sklearn.dummy import DummyClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

df = pd.read_csv("cleaned_data.csv").dropna()
print("Rows:", len(df))

# Stratify keeps the same class mix in train and test
X_train, X_test, y_train, y_test = train_test_split(
    df["text"],
    df["label"],
    test_size=0.2,
    random_state=42,
    stratify=df["label"],
)

# Dumb baseline: always predict the most common class 
dummy = DummyClassifier(strategy="most_frequent")
dummy.fit(np.zeros((len(X_train), 1)), y_train)
dummy_pred = dummy.predict(np.zeros((len(X_test), 1)))
print("\n--- Dumb baseline (always Neutral) ---")
print("Accuracy :", round(accuracy_score(y_test, dummy_pred), 3))
print("Macro F1 :", round(f1_score(y_test, dummy_pred, average="macro"), 3))

# TF-IDF + Logistic Regression 
model = Pipeline(
    [
        (
            "tfidf",
            TfidfVectorizer(ngram_range=(1, 2), min_df=2, sublinear_tf=True),
        ),
        ("clf", LogisticRegression(max_iter=1000, class_weight="balanced")),
    ]
)
model.fit(X_train, y_train)
pred = model.predict(X_test)

print("\n--- TF-IDF + Logistic Regression ---")
print("Accuracy :", round(accuracy_score(y_test, pred), 3))
print("Macro F1 :", round(f1_score(y_test, pred, average="macro"), 3))
print("\n", classification_report(y_test, pred))

labels = ["Negative", "Neutral", "Positive"]
print("Confusion matrix (rows = true, columns = predicted):")
print(pd.DataFrame(confusion_matrix(y_test, pred, labels=labels), index=labels, columns=labels))