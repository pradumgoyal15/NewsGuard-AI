import pandas as pd

from sklearn.dummy import DummyClassifier
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix
)


# ==================================================
# Configuration
# ==================================================

TRAIN_PATH = "data/train.csv"
TEST_PATH = "data/test.csv"


# ==================================================
# Load data
# ==================================================

print("Loading training data...")

train_df = pd.read_csv(
    TRAIN_PATH,
    keep_default_na=False
)

print("Loading testing data...")

test_df = pd.read_csv(
    TEST_PATH,
    keep_default_na=False
)


# ==================================================
# Separate features and labels
# ==================================================

X_train = train_df["text"]
y_train = train_df["label"]

X_test = test_df["text"]
y_test = test_df["label"]


# ==================================================
# Create Dummy Model
# ==================================================

print("\nCreating baseline model...")

model = DummyClassifier(
    strategy="most_frequent"
)


# ==================================================
# Train
# ==================================================

print("Training baseline model...")

model.fit(
    X_train,
    y_train
)


# ==================================================
# Predict
# ==================================================

print("Making predictions...")

y_pred = model.predict(X_test)


# ==================================================
# Evaluation
# ==================================================

accuracy = accuracy_score(
    y_test,
    y_pred
)


print("\n========== BASELINE RESULTS ==========")

print(
    f"Accuracy: {accuracy:.4f}"
)

print(
    f"Accuracy: {accuracy * 100:.2f}%"
)


print("\n========== CLASSIFICATION REPORT ==========")

print(
    classification_report(
        y_test,
        y_pred,
        target_names=[
            "Fake",
            "Real"
        ]
    )
)


print("\n========== CONFUSION MATRIX ==========")

print(
    confusion_matrix(
        y_test,
        y_pred
    )
)