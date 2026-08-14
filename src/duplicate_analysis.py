import pandas as pd

# Load dataset
df = pd.read_csv("data/WELFake_Dataset.csv")

print("\n========== ORIGINAL DATASET ==========")
print(f"Rows: {len(df)}")


# --------------------------------------------------
# 1. Analyze duplicate title + text combinations
# --------------------------------------------------

duplicate_mask = df.duplicated(
    subset=["title", "text"],
    keep=False
)

duplicates = df[duplicate_mask].copy()

print("\n========== DUPLICATE ARTICLE ROWS ==========")
print(f"Rows involved in duplicate groups: {len(duplicates)}")


# --------------------------------------------------
# 2. Number of unique duplicate groups
# --------------------------------------------------

duplicate_groups = (
    duplicates
    .groupby(["title", "text"], dropna=False)
    .size()
    .reset_index(name="count")
)

print("\n========== DUPLICATE GROUPS ==========")
print(f"Number of duplicate groups: {len(duplicate_groups)}")


# --------------------------------------------------
# 3. Check whether duplicate articles have
#    conflicting labels
# --------------------------------------------------

label_counts = (
    duplicates
    .groupby(["title", "text"], dropna=False)["label"]
    .nunique()
)

conflicting_groups = label_counts[label_counts > 1]

print("\n========== LABEL CONFLICTS ==========")
print(
    f"Duplicate article groups with conflicting labels: "
    f"{len(conflicting_groups)}"
)


# --------------------------------------------------
# 4. Show examples of duplicate articles
# --------------------------------------------------

print("\n========== DUPLICATE EXAMPLES ==========")

example_duplicates = (
    duplicates
    .sort_values(["title", "text"])
    [["title", "text", "label"]]
    .head(10)
)

print(example_duplicates.to_string(index=False))


# --------------------------------------------------
# 5. Show conflicting examples
# --------------------------------------------------

if len(conflicting_groups) > 0:

    print("\n========== CONFLICTING LABEL EXAMPLES ==========")

    conflicting_index = conflicting_groups.index

    conflicting_rows = (
        duplicates
        .set_index(["title", "text"])
        .loc[conflicting_index]
        [["label"]]
        .reset_index()
    )

    print(
        conflicting_rows
        .head(10)
        .to_string(index=False)
    )

else:

    print("\nNo conflicting duplicate labels found.")


# --------------------------------------------------
# 6. Check rows where BOTH title and text are missing
# --------------------------------------------------

both_missing = df[
    df["title"].isna() &
    df["text"].isna()
]

print("\n========== BOTH TITLE AND TEXT MISSING ==========")
print(f"Rows with both missing: {len(both_missing)}")


# --------------------------------------------------
# 7. Check completely unusable text
# --------------------------------------------------

empty_or_missing_text = df[
    df["text"].isna() |
    (df["text"].fillna("").astype(str).str.strip() == "")
]

print("\n========== UNUSABLE ARTICLE TEXT ==========")
print(
    f"Rows with missing OR empty text: "
    f"{len(empty_or_missing_text)}"
)


# --------------------------------------------------
# 8. Label distribution
# --------------------------------------------------

print("\n========== LABEL DISTRIBUTION ==========")
print(df["label"].value_counts().sort_index())


# --------------------------------------------------
# 9. Label distribution as percentages
# --------------------------------------------------

print("\n========== LABEL PERCENTAGE ==========")
print(
    (df["label"].value_counts(normalize=True)
     .sort_index() * 100)
    .round(2)
)