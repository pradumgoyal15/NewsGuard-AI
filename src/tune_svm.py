import pandas as pd

from sklearn.feature_extraction.text import TfidfVectorizer

from sklearn.svm import LinearSVC

from sklearn.model_selection import StratifiedKFold

from sklearn.metrics import accuracy_score


# ==================================================
# Configuration
# ==================================================

TRAIN_PATH = "data/train.csv"

MAX_FEATURES = 100000

C_VALUES = [
    0.1,
    0.5,
    1.0,
    2.0,
    5.0
]


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
# TF-IDF
# ==================================================

print("\nCreating TF-IDF vectorizer...")

vectorizer = TfidfVectorizer(
    lowercase=True,
    stop_words="english",
    ngram_range=(1, 2),
    max_features=MAX_FEATURES,
    sublinear_tf=True
)


print("Fitting TF-IDF...")

X_tfidf = vectorizer.fit_transform(X)


print(
    f"TF-IDF shape: {X_tfidf.shape}"
)


# ==================================================
# Cross-validation
# ==================================================

cv = StratifiedKFold(
    n_splits=5,
    shuffle=True,
    random_state=42
)


results = []


# ==================================================
# Test C values
# ==================================================

for C in C_VALUES:

    print("\n========================================")

    print(
        f"Testing C = {C}"
    )

    print("========================================")


    fold_scores = []


    for fold, (train_idx, val_idx) in enumerate(
        cv.split(X_tfidf, y),
        start=1
    ):

        print(
            f"Training fold {fold}/5..."
        )


        X_train = X_tfidf[train_idx]

        X_val = X_tfidf[val_idx]

        y_train = y.iloc[train_idx]

        y_val = y.iloc[val_idx]


        model = LinearSVC(
            C=C,
            random_state=42
        )


        model.fit(
            X_train,
            y_train
        )


        predictions = model.predict(
            X_val
        )


        score = accuracy_score(
            y_val,
            predictions
        )


        fold_scores.append(score)


        print(
            f"Fold {fold} accuracy: "
            f"{score:.4f}"
        )


    mean_score = sum(
        fold_scores
    ) / len(fold_scores)


    results.append(
        {
            "C": C,
            "mean_accuracy": mean_score
        }
    )


    print(
        f"\nMean accuracy for C={C}: "
        f"{mean_score:.4f}"
    )


# ==================================================
# Results
# ==================================================

results_df = pd.DataFrame(
    results
)


results_df = results_df.sort_values(
    "mean_accuracy",
    ascending=False
)


print("\n\n========================================")

print("CROSS-VALIDATION RESULTS")

print("========================================")

print(
    results_df.to_string(
        index=False
    )
)


# ==================================================
# Best parameter
# ==================================================

best_row = results_df.iloc[0]

best_C = best_row["C"]

best_score = best_row["mean_accuracy"]


print("\n========== BEST PARAMETER ==========")

print(
    f"Best C: {best_C}"
)

print(
    f"Mean CV Accuracy: "
    f"{best_score:.4f}"
)