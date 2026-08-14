import os
import joblib


# ============================================================
# CONFIGURATION
# ============================================================

# Get the project root directory
BASE_DIR = os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))
)

MODEL_PATH = os.path.join(
    BASE_DIR,
    "models",
    "newsguard_pipeline.pkl"
)


# ============================================================
# LOAD TRAINED MODEL
# ============================================================

def load_model():

    if not os.path.exists(MODEL_PATH):

        raise FileNotFoundError(
            f"Model not found at: {MODEL_PATH}"
        )

    return joblib.load(MODEL_PATH)


# ============================================================
# PREPARE ARTICLE
# ============================================================

def prepare_article(title, article_text):

    # Validate title
    if not isinstance(title, str):

        raise TypeError(
            "Title must be a string."
        )

    # Validate article text
    if not isinstance(article_text, str):

        raise TypeError(
            "Article text must be a string."
        )

    # Remove unnecessary spaces
    title = title.strip()
    article_text = article_text.strip()

    # Article cannot be empty
    if not article_text:

        raise ValueError(
            "Article text cannot be empty."
        )

    # Same format used during training
    combined_text = (
        title + " " + article_text
    ).strip()

    return combined_text


# ============================================================
# CALCULATE SIGNAL STRENGTH
# ============================================================

def calculate_signal_strength(decision_score):

    score = abs(decision_score)

    if score < 0.25:

        return "LOW"

    elif score < 0.75:

        return "MEDIUM"

    else:

        return "HIGH"


# ============================================================
# EXPLAIN MODEL SIGNAL
# ============================================================

def explain_signal(decision_score, prediction):

    score = abs(decision_score)

    if score < 0.25:

        return (
            "The model produced a weak signal. "
            "The article does not strongly resemble either "
            "class in the training data. Use this prediction "
            "with caution."
        )

    elif score < 0.75:

        if prediction == "REAL":

            return (
                "The model found a moderate signal associated "
                "with real-news examples in the training data."
            )

        else:

            return (
                "The model found a moderate signal associated "
                "with fake-news examples in the training data."
            )

    else:

        if prediction == "REAL":

            return (
                "The model found a strong signal associated "
                "with real-news examples in the training data."
            )

        else:

            return (
                "The model found a strong signal associated "
                "with fake-news examples in the training data."
            )


# ============================================================
# PREDICT NEWS
# ============================================================

def predict_news(title, article_text):

    # --------------------------------------------------------
    # Prepare input
    # --------------------------------------------------------

    combined_text = prepare_article(
        title,
        article_text
    )


    # --------------------------------------------------------
    # Load model
    # --------------------------------------------------------

    model = load_model()


    # --------------------------------------------------------
    # Make prediction
    # --------------------------------------------------------

    prediction = model.predict(
        [combined_text]
    )[0]


    # --------------------------------------------------------
    # Get SVM decision score
    # --------------------------------------------------------

    decision_score = float(
        model.decision_function(
            [combined_text]
        )[0]
    )


    # --------------------------------------------------------
    # Convert numeric label to readable label
    # --------------------------------------------------------

    if prediction == 1:

        label = "REAL"

    else:

        label = "FAKE"


    # --------------------------------------------------------
    # Signal strength
    # --------------------------------------------------------

    signal_strength = calculate_signal_strength(
        decision_score
    )


    # --------------------------------------------------------
    # Signal explanation
    # --------------------------------------------------------

    signal_explanation = explain_signal(
        decision_score,
        label
    )


    # --------------------------------------------------------
    # Article statistics
    # --------------------------------------------------------

    article_characters = len(
        article_text
    )

    article_words = len(
        article_text.split()
    )

    title_characters = len(
        title
    )


    # --------------------------------------------------------
    # Article quality warnings
    # --------------------------------------------------------

    warnings = []


    # Very short article
    if article_words < 20:

        warnings.append(
            "Article is very short. "
            "Prediction may be less reliable."
        )


    # Relatively short article
    elif article_words < 50:

        warnings.append(
            "Article is relatively short. "
            "Use the prediction with caution."
        )


    # Missing title
    if not title:

        warnings.append(
            "No headline was provided. "
            "Including the headline may improve analysis."
        )


    # Very little content
    if article_characters < 100:

        warnings.append(
            "Very little article content was provided."
        )


    # --------------------------------------------------------
    # Combine warnings
    # --------------------------------------------------------

    if warnings:

        warning = " ".join(warnings)

    else:

        warning = None


    # --------------------------------------------------------
    # RETURN RESULT
    # --------------------------------------------------------

    return {

        "prediction": label,

        "decision_score": round(
            decision_score,
            4
        ),

        "signal_strength": signal_strength,

        "signal_explanation": signal_explanation,

        "article_characters": article_characters,

        "article_words": article_words,

        "title_characters": title_characters,

        "warning": warning
    }


# ============================================================
# TEST PREDICTION ENGINE
# ============================================================

if __name__ == "__main__":

    print("=" * 60)

    print(
        "       NEWSGUARD AI — PREDICTION TEST"
    )

    print("=" * 60)


    # --------------------------------------------------------
    # Test title
    # --------------------------------------------------------

    test_title = (
        "Government announces new digital "
        "education policy"
    )


    # --------------------------------------------------------
    # Test article
    # --------------------------------------------------------

    test_article = """
    The government announced a new policy
    aimed at improving digital education and
    expanding access to technology in schools.
    The initiative will provide additional
    resources to educational institutions and
    support the development of digital learning
    programs across the country.
    """


    print("\n========== INPUT ==========")

    print(
        f"Title: {test_title}"
    )

    print(
        f"\nArticle:\n{test_article.strip()}"
    )


    try:

        result = predict_news(
            test_title,
            test_article
        )


        print("\n========== RESULT ==========")

        print(
            f"Prediction: "
            f"{result['prediction']}"
        )

        print(
            f"Decision score: "
            f"{result['decision_score']}"
        )

        print(
            f"Signal strength: "
            f"{result['signal_strength']}"
        )

        print(
            f"Signal explanation: "
            f"{result['signal_explanation']}"
        )

        print(
            f"Article characters: "
            f"{result['article_characters']}"
        )

        print(
            f"Article words: "
            f"{result['article_words']}"
        )

        print(
            f"Title characters: "
            f"{result['title_characters']}"
        )


        if result["warning"]:

            print(
                f"\n⚠️ Warning: "
                f"{result['warning']}"
            )

        else:

            print(
                "\n✓ Article length is "
                "suitable for analysis."
            )


    except Exception as error:

        print(
            f"\n❌ ERROR: {error}"
        )