import pandas as pd

from sklearn.feature_extraction.text import TfidfVectorizer

from sklearn.naive_bayes import MultinomialNB

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
# Load datasets
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


print("\n========== DATA ==========")

print(
    f"Training samples: {len(X_train)}"
)

print(
    f"Testing samples:  {len(X_test)}"
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

X_train_tfidf = vectorizer.fit_transform(
    X_train
)

print(
    f"Training TF-IDF shape: "
    f"{X_train_tfidf.shape}"
)


print("Transforming testing data...")

X_test_tfidf = vectorizer.transform(
    X_test
)

print(
    f"Testing TF-IDF shape: "
    f"{X_test_tfidf.shape}"
)


# ==================================================
# Create Naive Bayes model
# ==================================================

print("\nCreating Multinomial Naive Bayes model...")

model = MultinomialNB(
    alpha=0.1
)


# ==================================================
# Train
# ==================================================

print("Training model...")

model.fit(
    X_train_tfidf,
    y_train
)

print("Training complete!")


# ==================================================
# Predictions
# ==================================================

print("\nMaking predictions...")

y_pred = model.predict(
    X_test_tfidf
)

y_probability = model.predict_proba(
    X_test_tfidf
)[:, 1]


# ==================================================
# Accuracy
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
# Classification Report
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
# Confusion Matrix
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

roc_auc = roc_auc_score(
    y_test,
    y_probability
)


print("\n========== ROC-AUC ==========")

print(
    f"ROC-AUC: {roc_auc:.4f}"
)


# ==================================================
# Feature information
# ==================================================

print("\n========== TF-IDF INFORMATION ==========")

print(
    f"Number of features: "
    f"{len(vectorizer.get_feature_names_out())}"
)