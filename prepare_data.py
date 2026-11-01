import re

import pandas as pd

# The file has no header row; keep only the first two columns
df = pd.read_csv(
    "Roman Urdu DataSet.csv",
    header=None,
    usecols=[0, 1],
    names=["text", "label"],
    encoding="utf-8",
)
print("Rows loaded:", len(df))

# Fix labels 
df["label"] = df["label"].astype(str).str.strip().str.title()
df["label"] = df["label"].replace({"Neative": "Negative"})
df = df[df["label"].isin(["Positive", "Negative", "Neutral"])]

# Drop missing text
df = df.dropna(subset=["text"])

# Clean the text 
def clean_text(text):
    text = str(text).lower()
    text = re.sub(r"http\S+|www\.\S+", " ", text)  # links
    text = re.sub(r"@\w+", " ", text)  # mentions
    text = re.sub(r"[^\w\s]", " ", text)  # punctuation and emojis
    text = text.replace("_", " ")
    text = re.sub(r"([a-z])\1{2,}", r"\1\1", text)  # letters only, keeps numbers like 1999
    text = re.sub(r"\s+", " ", text).strip()
    return text


df["raw_text"] = df["text"]
df["text"] = df["text"].apply(clean_text)
df = df[df["text"].str.len() > 0]

#Remove duplicates and contradictions 
before = len(df)
label_counts = df.groupby("text")["label"].nunique()
conflicting = label_counts[label_counts > 1].index
print("Texts with conflicting labels (removed):", len(conflicting))
df = df[~df["text"].isin(conflicting)]
df = df.drop_duplicates(subset="text")
print("Duplicates and conflicts removed:", before - len(df))

# Report 
print("\nRows after cleaning:", len(df))
print("\nClass counts:")
print(df["label"].value_counts())
print("\nWords per text:")
print(df["text"].str.split().str.len().describe())

print("\nRandom examples (raw -> clean):")
for _, row in df.sample(8, random_state=1).iterrows():
    print(f"[{row['label']}] {str(row['raw_text'])[:80]}")
    print(f"        -> {row['text'][:80]}")

df[["text", "label"]].to_csv("cleaned_data.csv", index=False)
print("\nSaved cleaned_data.csv")