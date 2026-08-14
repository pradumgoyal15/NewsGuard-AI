import pandas as pd


# ==================================================
# Configuration
# ==================================================

TRAIN_PATH = "data/train.csv"
TEST_PATH = "data/test.csv"


# ==================================================
# Load datasets
# ==================================================

print("Loading training dataset...")

train_df = pd.read_csv(
    TRAIN_PATH,
    keep_default_na=False
)

print("Loading testing dataset...")

test_df = pd.read_csv(
    TEST_PATH,
    keep_default_na=False
)


# ==================================================
# Basic information
# ==================================================

print("\n========== DATASET SIZES ==========")

print(
    f"Training samples: {len(train_df)}"
)

print(
    f"Testing samples:  {len(test_df)}"
)


# ==================================================
# Exact text overlap
# ==================================================

print("\n========== EXACT TEXT OVERLAP ==========")

train_texts = set(
    train_df["text"]
)

test_texts = set(
    test_df["text"]
)

overlap = train_texts.intersection(
    test_texts
)

print(
    f"Exact duplicate texts across "
    f"train/test: {len(overlap)}"
)


# ==================================================
# Exact title + text overlap
# ==================================================

print("\n========== TITLE + TEXT OVERLAP ==========")

train_combined = set(
    train_df["text"]
)

test_combined = set(
    test_df["text"]
)

combined_overlap = train_combined.intersection(
    test_combined
)

print(
    f"Duplicate articles across "
    f"train/test: {len(combined_overlap)}"
)


# ==================================================
# Duplicate counts within datasets
# ==================================================

print("\n========== TRAINING DUPLICATES ==========")

print(
    f"Duplicate training texts: "
    f"{train_df['text'].duplicated().sum()}"
)


print("\n========== TESTING DUPLICATES ==========")

print(
    f"Duplicate testing texts: "
    f"{test_df['text'].duplicated().sum()}"
)


# ==================================================
# Empty text check
# ==================================================

print("\n========== EMPTY TEXT CHECK ==========")

print(
    f"Empty training texts: "
    f"{(train_df['text'].str.strip() == '').sum()}"
)

print(
    f"Empty testing texts: "
    f"{(test_df['text'].str.strip() == '').sum()}"
)


# ==================================================
# Final conclusion
# ==================================================

print("\n========== LEAKAGE CHECK COMPLETE ==========")

if len(overlap) == 0:

    print(
        "✅ No exact text overlap detected "
        "between training and testing data."
    )

else:

    print(
        "⚠️ Potential leakage detected."
    )