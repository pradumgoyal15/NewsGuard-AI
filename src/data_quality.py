import pandas as pd

# Load dataset
df = pd.read_csv("data/WELFake_Dataset.csv")

print("\n========== DATASET SIZE ==========")
print(f"Rows: {len(df)}")
print(f"Columns: {len(df.columns)}")


# --------------------------------------------------
# 1. Missing titles
# --------------------------------------------------

missing_title = df[df["title"].isna()]

print("\n========== MISSING TITLES ==========")
print(f"Number of missing titles: {len(missing_title)}")

print("\nExamples of articles with missing titles:")
print(missing_title[["text", "label"]].head(3).to_string())


# --------------------------------------------------
# 2. Missing article text
# --------------------------------------------------

missing_text = df[df["text"].isna()]

print("\n========== MISSING ARTICLE TEXT ==========")
print(f"Number of missing article texts: {len(missing_text)}")

print("\nExamples of articles with missing text:")
print(missing_text[["title", "label"]].to_string())


# --------------------------------------------------
# 3. Empty titles
# --------------------------------------------------

empty_title = df[
    df["title"].notna() &
    (df["title"].astype(str).str.strip() == "")
]

print("\n========== EMPTY TITLES ==========")
print(f"Number of empty titles: {len(empty_title)}")


# --------------------------------------------------
# 4. Empty article text
# --------------------------------------------------

empty_text = df[
    df["text"].notna() &
    (df["text"].astype(str).str.strip() == "")
]

print("\n========== EMPTY ARTICLE TEXT ==========")
print(f"Number of empty article texts: {len(empty_text)}")


# --------------------------------------------------
# 5. Content duplicates
# --------------------------------------------------

content_duplicates = df.duplicated(
    subset=["title", "text"]
).sum()

print("\n========== CONTENT DUPLICATES ==========")
print(f"Duplicate title + text combinations: {content_duplicates}")


# --------------------------------------------------
# 6. Label distribution
# --------------------------------------------------

print("\n========== LABEL DISTRIBUTION ==========")
print(df["label"].value_counts())

print("\n========== LABEL PERCENTAGE ==========")
print(
    (df["label"].value_counts(normalize=True) * 100)
    .round(2)
)