import pandas as pd

from sklearn.feature_extraction.text import TfidfVectorizer

from sklearn.linear_model import LogisticRegression

from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    roc_auc_score
)


# ==================================================
# Configuration
# ==================================================

TRAIN_PATH = "data/train.csv"
TEST_PATH = "data/test.csv"

MAX_FEATURES = 100000


# ==================================================
# 1. Load datasets
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
# 2. Separate features and labels
# ==================================================

X_train = train_df["text"]

y_train = train_df["label"]

X_test = test_df["text"]

y_test = test_df["label"]


print("\n========== DATA ==========")

print(f"Training samples: {len(X_train)}")

print(f"Testing samples:  {len(X_test)}")


# ==================================================
# 3. Create TF-IDF Vectorizer
# ==================================================

print("\nCreating TF-IDF vectorizer...")

vectorizer = TfidfVectorizer(
    lowercase=True,
    stop_words="english",
    ngram_range=(1, 2),
    max_features=MAX_FEATURES,
    sublinear_tf=True
)


# ==================================================
# 4. Fit TF-IDF ONLY on training data
# ==================================================

print("Fitting TF-IDF on training data...")

X_train_tfidf = vectorizer.fit_transform(
    X_train
)


print(
    f"Training TF-IDF shape: "
    f"{X_train_tfidf.shape}"
)


# ==================================================
# 5. Transform test data
# ==================================================

print("Transforming testing data...")

X_test_tfidf = vectorizer.transform(
    X_test
)


print(
    f"Testing TF-IDF shape: "
    f"{X_test_tfidf.shape}"
)


# ==================================================
# 6. Create Logistic Regression model
# ==================================================

print("\nCreating Logistic Regression model...")

model = LogisticRegression(
    max_iter=1000,
    C=2.0,
    random_state=42
)


# ==================================================
# 7. Train model
# ==================================================

print("Training model...")

model.fit(
    X_train_tfidf,
    y_train
)


print("Training complete!")


# ==================================================
# 8. Make predictions
# ==================================================

print("\nMaking predictions...")

y_pred = model.predict(
    X_test_tfidf
)

y_probability = model.predict_proba(
    X_test_tfidf
)[:, 1]


# ==================================================
# 9. Accuracy
# ==================================================

accuracy = accuracy_score(
    y_test,
    y_pred
)


print("\n========== MODEL RESULTS ==========")

print(
    f"Accuracy: {accuracy:.4f}"
)

print(
    f"Accuracy: {accuracy * 100:.2f}%"
)


# ==================================================
# 10. Classification Report
# ==================================================

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


# ==================================================
# 11. Confusion Matrix
# ==================================================

print("\n========== CONFUSION MATRIX ==========")

cm = confusion_matrix(
    y_test,
    y_pred
)

print(cm)


# ==================================================
# 12. ROC-AUC
# ==================================================

roc_auc = roc_auc_score(
    y_test,
    y_probability
)


print("\n========== ROC-AUC ==========")

print(
    f"ROC-AUC: {roc_auc:.4f}"
)


# ==================================================
# 13. Model vocabulary
# ==================================================

print("\n========== TF-IDF INFORMATION ==========")

print(
    f"Number of features: "
    f"{len(vectorizer.get_feature_names_out())}"
)