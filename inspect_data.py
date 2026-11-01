import pandas as pd

# header=None because many versions of this file have no header row
df = pd.read_csv("Roman Urdu DataSet.csv", header=None, encoding="utf-8")

print("Shape:", df.shape)
print("\nFirst 10 rows:")
print(df.head(10).to_string())

print("\nUnique values per column:")
print(df.nunique())

print("\nMissing values per column:")
print(df.isna().sum())

for col in df.columns:
    if df[col].nunique() <= 5:
        print(f"\nValue counts for column {col}:")
        print(df[col].value_counts(dropna=False))