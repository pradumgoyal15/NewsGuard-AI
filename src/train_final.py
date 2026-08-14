import os
import time
import pandas as pd

from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC

import joblib


# ==================================================
# Configuration
# ==================================================

TRAIN_PATH = "data/train.csv"
MODEL_PATH = "models/newsguard_pipeline.pkl"

MAX_FEATURES = 100000
SVM_C = 1.0


# ==================================================
# Start timer
# ==================================================

start_time = time.time()


print("=" * 60)
print("        NEWSGUARD AI — FINAL MODEL TRAINING")
print("=" * 60)


# ==================================================
# Check training dataset
# ==================================================

if not os.path.exists(TRAIN_PATH):

    raise FileNotFoundError(
        f"Training dataset not found: {TRAIN_PATH}"
    )


# ==================================================
# Create models directory
# ==================================================

os.makedirs(
    "models",
    exist_ok=True
)


# ==================================================
# Load training data
# ==================================================

print("\nLoading training dataset...")

train_df = pd.read_csv(
    TRAIN_PATH,
    keep_default_na=False
)


print(
    f"Training samples: {len(train_df)}"
)


# ==================================================
# Check required columns
# ==================================================

required_columns = [
    "text",
    "label"
]


for column in required_columns:

    if column not in train_df.columns:

        raise ValueError(
            f"Required column missing: {column}"
        )


# ==================================================
# Prepare training data
# ==================================================

X_train = train_df["text"]

y_train = train_df["label"]


print("\n========== LABEL DISTRIBUTION ==========")

print(
    y_train.value_counts()
)


# ==================================================
# Create final pipeline
# ==================================================

print("\nCreating final ML pipeline...")


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
                C=SVM_C,
                random_state=42
            )
        )
    ]
)


print("\n========== MODEL CONFIGURATION ==========")

print(
    f"TF-IDF max features: {MAX_FEATURES}"
)

print(
    "TF-IDF n-gram range: (1, 2)"
)

print(
    "TF-IDF lowercase: True"
)

print(
    "TF-IDF stop words: English"
)

print(
    "TF-IDF sublinear TF: True"
)

print(
    f"Linear SVM C: {SVM_C}"
)


# ==================================================
# Train final model
# ==================================================

print("\n========== TRAINING ==========")

print(
    "Training final model..."
)

pipeline.fit(
    X_train,
    y_train
)

print(
    "Training completed successfully!"
)


# ==================================================
# Inspect vocabulary
# ==================================================

tfidf = pipeline.named_steps["tfidf"]

feature_count = len(
    tfidf.get_feature_names_out()
)


print("\n========== TF-IDF INFORMATION ==========")

print(
    f"Learned features: {feature_count}"
)


# ==================================================
# Save model
# ==================================================

print("\n========== SAVING MODEL ==========")

joblib.dump(
    pipeline,
    MODEL_PATH
)


print(
    f"Model saved to: {MODEL_PATH}"
)


# ==================================================
# Verify saved model
# ==================================================

if os.path.exists(MODEL_PATH):

    file_size = os.path.getsize(
        MODEL_PATH
    )

    file_size_mb = file_size / (
        1024 * 1024
    )

    print(
        f"Model file size: "
        f"{file_size_mb:.2f} MB"
    )

else:

    raise RuntimeError(
        "Model file was not created."
    )


# ==================================================
# Finish
# ==================================================

elapsed_time = (
    time.time() - start_time
)


print("\n" + "=" * 60)

print(
    "FINAL MODEL TRAINING COMPLETE"
)

print("=" * 60)

print(
    f"Training time: "
    f"{elapsed_time:.2f} seconds"
)

print(
    f"Saved model: {MODEL_PATH}"
)