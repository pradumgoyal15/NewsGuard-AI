import pandas as pd

from sklearn.pipeline import Pipeline

from sklearn.feature_extraction.text import TfidfVectorizer

from sklearn.svm import LinearSVC

from sklearn.model_selection import (
    StratifiedKFold,
    cross_validate
)


# ==================================================
# Configuration
# ==================================================

TRAIN_PATH = "data/train.csv"

MAX_FEATURES = 100000

C = 1.0


# ==================================================
# Load training data
# ==================================================

print("Loading training dataset...")

train_df = pd.read_csv(
    TRAIN_PATH,
    keep_default_na=False
)


X = train_df["text"]

y = train_df["label"]


print("\n========== DATA ==========")

print(
    f"Training samples: {len(X)}"
)


# ==================================================
# Create Pipeline
# ==================================================

print("\nCreating ML pipeline...")

pipeline = Pipeline(
    [
        (
            "tfidf",
            TfidfVectorizer(
                lowercase=True,
                stop_words="english",
                ngram_range=(1, 2),
                max_features=MAX_FEATURES,
                sublinear_tf=True
            )
        ),
        (
            "svm",
            LinearSVC(
                C=C,
                random_state=42
            )
        )
    ]
)


# ==================================================
# Cross-validation
# ==================================================

cv = StratifiedKFold(
    n_splits=5,
    shuffle=True,
    random_state=42
)


print("\n========== CROSS-VALIDATION ==========")

print("Running 5-fold cross-validation...")

scores = cross_validate(
    pipeline,
    X,
    y,
    cv=cv,
    scoring=[
        "accuracy",
        "precision_macro",
        "recall_macro",
        "f1_macro"
    ],
    n_jobs=-1,
    return_train_score=False
)


# ==================================================
# Results
# ==================================================

print("\n========== FOLD RESULTS ==========")

for i, score in enumerate(
    scores["test_accuracy"],
    start=1
):

    print(
        f"Fold {i} accuracy: "
        f"{score:.4f}"
    )


# ==================================================
# Mean scores
# ==================================================

print("\n========== FINAL CV RESULTS ==========")

print(
    f"Mean Accuracy: "
    f"{scores['test_accuracy'].mean():.4f}"
)

print(
    f"Std Accuracy: "
    f"{scores['test_accuracy'].std():.4f}"
)

print(
    f"Mean Precision: "
    f"{scores['test_precision_macro'].mean():.4f}"
)

print(
    f"Mean Recall: "
    f"{scores['test_recall_macro'].mean():.4f}"
)

print(
    f"Mean F1: "
    f"{scores['test_f1_macro'].mean():.4f}"
)


# ==================================================
# Configuration summary
# ==================================================

print("\n========== CONFIGURATION ==========")

print(
    f"TF-IDF features: {MAX_FEATURES}"
)

print(
    "N-gram range: (1, 2)"
)

print(
    f"SVM C: {C}"
)

print(
    "Cross-validation folds: 5"
)