import pandas as pd

INPUT_PATH = "data/WELFake_Dataset.csv"
OUTPUT_PATH = "data/WELFake_cleaned.csv"


print("Loading dataset...")

df = pd.read_csv(INPUT_PATH)

print(f"Original rows: {len(df)}")


# --------------------------------------------------
# 1. Remove unnecessary index column
# --------------------------------------------------

if "Unnamed: 0" in df.columns:
    df = df.drop(columns=["Unnamed: 0"])

print(f"After removing index column: {len(df)} rows")


# --------------------------------------------------
# 2. Replace missing titles with empty strings
# --------------------------------------------------

df["title"] = df["title"].fillna("")


# --------------------------------------------------
# 3. Remove missing/empty article text
# --------------------------------------------------

df["text"] = df["text"].fillna("")

df["text"] = df["text"].astype(str)

df["text"] = df["text"].str.strip()

before_text_cleaning = len(df)

df = df[df["text"] != ""].copy()

removed_empty_text = (
    before_text_cleaning - len(df)
)

print(
    f"Removed rows with empty article text: "
    f"{removed_empty_text}"
)


# --------------------------------------------------
# 4. Remove duplicate title + text combinations
# --------------------------------------------------

before_duplicates = len(df)

df = df.drop_duplicates(
    subset=["title", "text"],
    keep="first"
).copy()

removed_duplicates = (
    before_duplicates - len(df)
)

print(
    f"Removed duplicate articles: "
    f"{removed_duplicates}"
)


# --------------------------------------------------
# 5. Reset DataFrame index
# --------------------------------------------------

df = df.reset_index(drop=True)


# --------------------------------------------------
# 6. Final dataset information
# --------------------------------------------------

print("\n========== CLEANING SUMMARY ==========")

print(f"Original rows:  {72134}")
print(f"Final rows:     {len(df)}")
print(
    f"Total removed:  {72134 - len(df)}"
)


print("\n========== FINAL COLUMNS ==========")

print(df.columns.tolist())


print("\n========== FINAL MISSING VALUES ==========")

print(df.isnull().sum())


print("\n========== FINAL DUPLICATES ==========")

print(
    df.duplicated(
        subset=["title", "text"]
    ).sum()
)


print("\n========== FINAL LABEL DISTRIBUTION ==========")

print(df["label"].value_counts().sort_index())


# --------------------------------------------------
# 7. Save cleaned dataset
# --------------------------------------------------

df.to_csv(
    OUTPUT_PATH,
    index=False
)

print(
    f"\nCleaned dataset saved to: "
    f"{OUTPUT_PATH}"
)