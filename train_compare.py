import time

import joblib
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
)
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
from sklearn.naive_bayes import ComplementNB
from sklearn.pipeline import FeatureUnion, Pipeline
from sklearn.svm import LinearSVC

df = pd.read_csv("cleaned_data.csv").dropna()

# Same split as before, so results are comparable
X_train, X_test, y_train, y_test = train_test_split(
    df["text"],
    df["label"],
    test_size=0.2,
    random_state=42,
    stratify=df["label"],
)


def word_tfidf():
    return TfidfVectorizer(ngram_range=(1, 2), min_df=2, sublinear_tf=True)


def char_tfidf():
    # Character n-grams are robust to spelling variations
    return TfidfVectorizer(
        analyzer="char_wb", ngram_range=(2, 5), min_df=3, sublinear_tf=True
    )


def word_and_char():
    return FeatureUnion([("word", word_tfidf()), ("char", char_tfidf())])


def logreg():
    return LogisticRegression(max_iter=1000, class_weight="balanced")


candidates = {
    "Word TF-IDF + LogReg": Pipeline([("feat", word_tfidf()), ("clf", logreg())]),
    "Word TF-IDF + Naive Bayes": Pipeline([("feat", word_tfidf()), ("clf", ComplementNB())]),
    "Char n-grams + LogReg": Pipeline([("feat", char_tfidf()), ("clf", logreg())]),
    "Word+Char + LogReg": Pipeline([("feat", word_and_char()), ("clf", logreg())]),
    "Word+Char + LinearSVC": Pipeline(
        [("feat", word_and_char()), ("clf", LinearSVC(class_weight="balanced"))]
    ),
}

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

print("5-fold cross-validation on the training data (macro F1):\n")
results = {}
for name, pipe in candidates.items():
    start = time.time()
    scores = cross_val_score(pipe, X_train, y_train, cv=cv, scoring="f1_macro", n_jobs=-1)
    results[name] = scores.mean()
    print(f"{name:28s} {scores.mean():.3f} (+/- {scores.std():.3f})   [{time.time() - start:.0f}s]")

best_name = max(results, key=results.get)
print("\nBest by cross-validation:", best_name)

# Final check on the untouched test set, done once
best = candidates[best_name]
best.fit(X_train, y_train)
pred = best.predict(X_test)

print("\n--- Final test-set result ---")
print("Accuracy :", round(accuracy_score(y_test, pred), 3))
print("Macro F1 :", round(f1_score(y_test, pred, average="macro"), 3))
print("\n", classification_report(y_test, pred))

labels = ["Negative", "Neutral", "Positive"]
print("Confusion matrix (rows = true, columns = predicted):")
print(pd.DataFrame(confusion_matrix(y_test, pred, labels=labels), index=labels, columns=labels))

joblib.dump(best, "model.pkl")
print("\nSaved model.pkl")