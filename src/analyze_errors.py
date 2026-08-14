import pandas as pd

from sklearn.feature_extraction.text import TfidfVectorizer

from sklearn.linear_model import LogisticRegression


# ==================================================
# Configuration
# ==================================================

TRAIN_PATH = "data/train.csv"
TEST_PATH = "data/test.csv"

MAX_FEATURES = 100000


# ==================================================
# Load data
# ==================================================

print("Loading datasets...")

train_df = pd.read_csv(
    TRAIN_PATH,
    keep_default_na=False
)

test_df = pd.read_csv(
    TEST_PATH,
    keep_default_na=False
)


X_train = train_df["text"]
y_train = train_df["label"]

X_test = test_df["text"]
y_test = test_df["label"]


# ==================================================
# TF-IDF
# ==================================================

print("Creating TF-IDF...")

vectorizer = TfidfVectorizer(
    lowercase=True,
    stop_words="english",
    ngram_range=(1, 2),
    max_features=MAX_FEATURES,
    sublinear_tf=True
)

X_train_tfidf = vectorizer.fit_transform(
    X_train
)

X_test_tfidf = vectorizer.transform(
    X_test
)


# ==================================================
# Train model
# ==================================================

print("Training Logistic Regression...")

model = LogisticRegression(
    max_iter=1000,
    C=2.0,
    random_state=42
)

model.fit(
    X_train_tfidf,
    y_train
)


# ==================================================
# Predictions
# ==================================================

print("Generating predictions...")

predictions = model.predict(
    X_test_tfidf
)

probabilities = model.predict_proba(
    X_test_tfidf
)


# ==================================================
# Find errors
# ==================================================

test_results = test_df.copy()

test_results["predicted"] = predictions

test_results["fake_probability"] = probabilities[:, 0]

test_results["real_probability"] = probabilities[:, 1]


errors = test_results[
    test_results["label"] != test_results["predicted"]
].copy()


# ==================================================
# Error statistics
# ==================================================

print("\n========== ERROR ANALYSIS ==========")

print(
    f"Total test articles: {len(test_results)}"
)

print(
    f"Incorrect predictions: {len(errors)}"
)

print(
    f"Error rate: "
    f"{len(errors) / len(test_results) * 100:.2f}%"
)


# ==================================================
# False Positives
# Actual Real → Predicted Fake
# ==================================================

false_positives = errors[
    (errors["label"] == 1)
    &
    (errors["predicted"] == 0)
]

print("\n========== FALSE POSITIVES ==========")

print(
    f"Count: {len(false_positives)}"
)


# ==================================================
# False Negatives
# Actual Fake → Predicted Real
# ==================================================

false_negatives = errors[
    (errors["label"] == 0)
    &
    (errors["predicted"] == 1)
]

print("\n========== FALSE NEGATIVES ==========")

print(
    f"Count: {len(false_negatives)}"
)


# ==================================================
# Show examples
# ==================================================

print("\n========== SAMPLE FALSE POSITIVES ==========")

for i, (_, row) in enumerate(
    false_positives.head(5).iterrows(),
    start=1
):

    print("\n----------------------------------------")

    print(f"Example {i}")

    print("Actual label:", row["label"])

    print("Predicted:", row["predicted"])

    print(
        "Real probability:",
        round(row["real_probability"], 4)
    )

    print(
        "Text:",
        row["text"][:1000]
    )


print("\n========== SAMPLE FALSE NEGATIVES ==========")

for i, (_, row) in enumerate(
    false_negatives.head(5).iterrows(),
    start=1
):

    print("\n----------------------------------------")

    print(f"Example {i}")

    print("Actual label:", row["label"])

    print("Predicted:", row["predicted"])

    print(
        "Fake probability:",
        round(row["fake_probability"], 4)
    )

    print(
        "Text:",
        row["text"][:1000]
    )