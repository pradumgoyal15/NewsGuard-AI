import os
import joblib
import pandas as pd

from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    roc_auc_score
)


# ==================================================
# Configuration
# ==================================================

MODEL_PATH = "models/newsguard_pipeline.pkl"
TEST_PATH = "data/test.csv"


# ==================================================
# Header
# ==================================================

print("=" * 60)
print("       NEWSGUARD AI — PRODUCTION MODEL TEST")
print("=" * 60)


# ==================================================
# Check files
# ==================================================

if not os.path.exists(MODEL_PATH):

    raise FileNotFoundError(
        f"Model not found: {MODEL_PATH}"
    )


if not os.path.exists(TEST_PATH):

    raise FileNotFoundError(
        f"Test dataset not found: {TEST_PATH}"
    )


# ==================================================
# Load model
# ==================================================

print("\nLoading saved model...")

model = joblib.load(
    MODEL_PATH
)

print("Model loaded successfully!")


# ==================================================
# Load test dataset
# ==================================================

print("\nLoading test dataset...")

test_df = pd.read_csv(
    TEST_PATH,
    keep_default_na=False
)


print(
    f"Testing samples: {len(test_df)}"
)


# ==================================================
# Prepare data
# ==================================================

X_test = test_df["text"]

y_test = test_df["label"]


# ==================================================
# Make predictions
# ==================================================

print("\nMaking predictions...")

y_pred = model.predict(
    X_test
)


# ==================================================
# Accuracy
# ==================================================

accuracy = accuracy_score(
    y_test,
    y_pred
)


print("\n========== PRODUCTION MODEL RESULTS ==========")

print(
    f"Accuracy: {accuracy:.4f}"
)

print(
    f"Accuracy: {accuracy * 100:.2f}%"
)


# ==================================================
# Classification report
# ==================================================

print("\n========== CLASSIFICATION REPORT ==========")

print(
    classification_report(
        y_test,
        y_pred,
        target_names=[
            "Fake",
            "Real"
        ],
        zero_division=0
    )
)


# ==================================================
# Confusion matrix
# ==================================================

print("\n========== CONFUSION MATRIX ==========")

cm = confusion_matrix(
    y_test,
    y_pred
)

print(cm)


# ==================================================
# ROC-AUC
# ==================================================

print("\n========== ROC-AUC ==========")

decision_scores = model.decision_function(
    X_test
)

roc_auc = roc_auc_score(
    y_test,
    decision_scores
)

print(
    f"ROC-AUC: {roc_auc:.4f}"
)


# ==================================================
# Final verification
# ==================================================

print("\n========== VERIFICATION ==========")

if accuracy >= 0.95:

    print(
        "✅ Production model performance is excellent."
    )

else:

    print(
        "⚠️ Production model accuracy is below 95%."
    )


print("\n" + "=" * 60)

print(
    "PRODUCTION MODEL TEST COMPLETE"
)

print("=" * 60)