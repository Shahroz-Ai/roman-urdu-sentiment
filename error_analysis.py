import joblib
import pandas as pd
from sklearn.model_selection import train_test_split

df = pd.read_csv("cleaned_data.csv").dropna()

# Same split as training, so this is the same test set
X_train, X_test, y_train, y_test = train_test_split(
    df["text"],
    df["label"],
    test_size=0.2,
    random_state=42,
    stratify=df["label"],
)

model = joblib.load("model.pkl")
classes = model.named_steps["clf"].classes_

proba = model.predict_proba(X_test)
pred = classes[proba.argmax(axis=1)]
conf = proba.max(axis=1)

res = pd.DataFrame(
    {"text": X_test.values, "true": y_test.values, "pred": pred, "confidence": conf}
)
res["correct"] = res["true"] == res["pred"]

# ---- 1. Is the confidence score trustworthy? ----
print("Confidence vs accuracy:")
res["bucket"] = pd.cut(res["confidence"], bins=[0, 0.45, 0.55, 0.65, 0.75, 1.0])
print(res.groupby("bucket", observed=True)["correct"].agg(["count", "mean"]).round(3))

# ---- 2. Most confident wrong predictions ----
wrong = res[~res["correct"]].sort_values("confidence", ascending=False)
wrong.to_csv("errors.csv", index=False)

print("\nMost confident WRONG predictions (check if the label is the problem):")
for _, r in wrong.head(15).iterrows():
    print(f"true={r['true']:8s} pred={r['pred']:8s} conf={r['confidence']:.2f} | {r['text'][:90]}")

# ---- 3. Top words per class ----
features = model.named_steps["feat"].get_feature_names_out()
coefs = model.named_steps["clf"].coef_
word_idx = [i for i, f in enumerate(features) if f.startswith("word__")]

print("\nTop words per class:")
for ci, cls in enumerate(classes):
    top = sorted(word_idx, key=lambda i: coefs[ci][i], reverse=True)[:15]
    print(f"{cls}: {[features[i].replace('word__', '') for i in top]}")