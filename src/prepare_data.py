import pandas as pd

from sklearn.model_selection import train_test_split


# ==================================================
# Configuration
# ==================================================

DATA_PATH = "data/WELFake_cleaned.csv"

TRAIN_PATH = "data/train.csv"
TEST_PATH = "data/test.csv"

RANDOM_STATE = 42
TEST_SIZE = 0.20


# ==================================================
# 1. Load cleaned dataset
# ==================================================

print("Loading cleaned dataset...")

df = pd.read_csv(
    DATA_PATH,
    keep_default_na=False
)

print(f"Total articles: {len(df)}")


# ==================================================
# 2. Create combined text
# ==================================================

print("\nCreating combined text...")

df["combined_text"] = (
    df["title"].str.strip()
    + " "
    + df["text"].str.strip()
).str.strip()


# ==================================================
# 3. Separate features and labels
# ==================================================

X = df["combined_text"]

y = df["label"]


# ==================================================
# 4. Train/Test Split
# ==================================================

print("\nSplitting dataset...")

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=TEST_SIZE,
    random_state=RANDOM_STATE,
    stratify=y
)


# ==================================================
# 5. Create DataFrames
# ==================================================

train_df = pd.DataFrame({
    "text": X_train,
    "label": y_train
})

test_df = pd.DataFrame({
    "text": X_test,
    "label": y_test
})


# ==================================================
# 6. Reset indexes
# ==================================================

train_df = train_df.reset_index(drop=True)

test_df = test_df.reset_index(drop=True)


# ==================================================
# 7. Display statistics
# ==================================================

print("\n========== SPLIT RESULTS ==========")

print(f"Training samples: {len(train_df)}")
print(f"Testing samples:  {len(test_df)}")


print("\n========== TRAINING LABEL DISTRIBUTION ==========")

print(
    train_df["label"]
    .value_counts()
    .sort_index()
)


print("\n========== TESTING LABEL DISTRIBUTION ==========")

print(
    test_df["label"]
    .value_counts()
    .sort_index()
)


print("\n========== TRAINING LABEL PERCENTAGE ==========")

print(
    (
        train_df["label"]
        .value_counts(normalize=True)
        .sort_index()
        * 100
    ).round(2)
)


print("\n========== TESTING LABEL PERCENTAGE ==========")

print(
    (
        test_df["label"]
        .value_counts(normalize=True)
        .sort_index()
        * 100
    ).round(2)
)


# ==================================================
# 8. Show examples
# ==================================================

print("\n========== TRAINING EXAMPLE ==========")

print(train_df.iloc[0]["text"][:1000])


print("\n========== LABEL ==========")

print(train_df.iloc[0]["label"])


# ==================================================
# 9. Save datasets
# ==================================================

print("\nSaving datasets...")

train_df.to_csv(
    TRAIN_PATH,
    index=False
)

test_df.to_csv(
    TEST_PATH,
    index=False
)


print("\n========== COMPLETE ==========")

print(f"Training data saved to: {TRAIN_PATH}")

print(f"Testing data saved to:  {TEST_PATH}")