import streamlit as st
import sys
import os


# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="NewsGuard AI",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded"
)


# ============================================================
# IMPORT PREDICTION ENGINE
# ============================================================

SRC_PATH = os.path.join(
    os.path.dirname(__file__),
    "src"
)

if SRC_PATH not in sys.path:
    sys.path.append(SRC_PATH)

from predictor import predict_news


# ============================================================
# SESSION STATE
# ============================================================

if "analysis_result" not in st.session_state:
    st.session_state.analysis_result = None

if "title_input" not in st.session_state:
    st.session_state.title_input = ""

if "article_input" not in st.session_state:
    st.session_state.article_input = ""


# ============================================================
# SAMPLE ARTICLE
# ============================================================

SAMPLE_TITLE = (
    "Government announces new digital education policy"
)

SAMPLE_ARTICLE = """
The government announced a new policy aimed at improving
digital education and expanding access to technology in
schools. The initiative will provide additional resources
to educational institutions and support digital learning
programs across the country.
"""


# ============================================================
# CALLBACK FUNCTIONS
# ============================================================

def load_sample():
    """Load the sample article into the input fields."""

    st.session_state.title_input = SAMPLE_TITLE
    st.session_state.article_input = SAMPLE_ARTICLE.strip()

    # Clear any previous result
    st.session_state.analysis_result = None


def clear_inputs():
    """Clear all input fields and previous results."""

    st.session_state.title_input = ""
    st.session_state.article_input = ""
    st.session_state.analysis_result = None


# ============================================================
# CUSTOM CSS
# ============================================================

st.markdown(
    """
    <style>
    /* ======================================================
       GLOBAL
       ====================================================== */
    .stApp {
        background: #0b0f17;
    }
    .main {
        padding-top: 1.5rem;
    }
    /* Hide unnecessary Streamlit elements */
    #MainMenu {
        visibility: hidden;
    }
    footer {
        visibility: hidden;
    }
    /* ======================================================
       HERO
       ====================================================== */
    .hero {
        padding: 2.5rem 2.8rem;
        border-radius: 22px;
        margin-bottom: 2rem;
        background:
            radial-gradient(
                circle at top right,
                rgba(59, 130, 246, 0.16),
                transparent 35%
            ),
            linear-gradient(
                135deg,
                #151b28,
                #0f141e
            );
        border: 1px solid #293449;

        box-shadow:
            0 12px 40px rgba(0, 0, 0, 0.25);
    }
    .hero-badge {
        display: inline-block;
        padding: 0.35rem 0.8rem;
        border-radius: 999px;
        background: rgba(59, 130, 246, 0.12);
        border: 1px solid rgba(59, 130, 246, 0.3);
        color: #7db5ff;
        font-size: 0.78rem;
        font-weight: 600;
        margin-bottom: 1rem;
    }
    .hero-title {
        font-size: 3rem;
        font-weight: 800;
        line-height: 1.1;
        color: #f4f7fb;
        margin-bottom: 0.5rem;
    }
    .hero-subtitle {
        font-size: 1.2rem;
        font-weight: 600;
        color: #9fb7d5;
        margin-bottom: 0.7rem;
    }
    .hero-description {
        max-width: 800px;
        color: #8996aa;
        font-size: 0.95rem;
        line-height: 1.6;
    }
    /* ======================================================
       SECTION HEADINGS
       ====================================================== */
    .section-title {
        font-size: 1.45rem;
        font-weight: 750;
        color: #f1f5f9;
        margin-top: 1.8rem;
        margin-bottom: 0.4rem;
    }
    .section-description {
        color: #8996aa;
        margin-bottom: 1rem;
    }
    /* ======================================================
       RESULT CARDS
       ====================================================== */
    .result-real {
        padding: 1.8rem 2rem;
        border-radius: 18px;
        background:
            linear-gradient(
                135deg,
                rgba(16, 185, 129, 0.16),
                rgba(6, 78, 59, 0.18)
            );
        border: 1px solid rgba(52, 211, 153, 0.35);
        box-shadow:
            0 10px 35px rgba(16, 185, 129, 0.06);
        margin: 1rem 0;
    }
    .result-fake {
        padding: 1.8rem 2rem;
        border-radius: 18px;
        background:
            linear-gradient(
                135deg,
                rgba(239, 68, 68, 0.16),
                rgba(127, 29, 29, 0.18)
            );
        border: 1px solid rgba(248, 113, 113, 0.35);
        box-shadow:
            0 10px 35px rgba(239, 68, 68, 0.06);
        margin: 1rem 0;
    }
    .result-title {
        font-size: 2rem;
        font-weight: 800;
        color: #f8fafc;
        margin-bottom: 0.4rem;
    }
    .result-description {
        color: #aeb9c9;
        line-height: 1.6;
    }
    /* ======================================================
       METRIC CARDS
       ====================================================== */
    .metric-card {
        padding: 1.3rem;
        border-radius: 16px;
        background: #131923;
        border: 1px solid #293449;
        min-height: 100px;
    }
    .metric-label {
        color: #7f8da3;
        font-size: 0.8rem;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.04em;
    }
    .metric-value {
        color: #f4f7fb;
        font-size: 1.7rem;
        font-weight: 750;
        margin-top: 0.35rem;
    }
    /* ======================================================
       SIGNAL BAR
       ====================================================== */
    .signal-container {
        margin: 1rem 0 1.5rem 0;
    }
    .signal-label {
        display: flex;
        justify-content: space-between;
        color: #aab5c5;
        font-size: 0.85rem;
        margin-bottom: 0.4rem;
    }
    .signal-track {
        width: 100%;
        height: 9px;
        background: #202735;
        border-radius: 999px;
        overflow: hidden;
    }
    .signal-fill-low {
        height: 100%;
        width: 25%;
        background: #f59e0b;
        border-radius: 999px;
    }
    .signal-fill-medium {
        height: 100%;
        width: 55%;
        background: #3b82f6;
        border-radius: 999px;
    }
    .signal-fill-high {
        height: 100%;
        width: 90%;
        background: #10b981;
        border-radius: 999px;
    }
    /* ======================================================
       ABOUT CARDS
       ====================================================== */
    .about-card {
        padding: 1.4rem;
        border-radius: 16px;
        background: #121720;
        border: 1px solid #293449;
        height: 100%;
    }
    .about-icon {
        font-size: 1.5rem;
        margin-bottom: 0.5rem;
    }
    .about-title {
        color: #f1f5f9;
        font-weight: 700;
        font-size: 1rem;
        margin-bottom: 0.4rem;
    }
    .about-text {
        color: #8794a8;
        font-size: 0.85rem;
        line-height: 1.55;
    }
    /* ======================================================
       FOOTER
       ====================================================== */
    .footer {
        text-align: center;
        color: #657186;
        padding: 2.5rem 0 1rem;
        font-size: 0.82rem;
    }
    /* ======================================================
       SIDEBAR
       ====================================================== */
    [data-testid="stSidebar"] {
        background: #10141d;
        border-right: 1px solid #242c3a;
    }
    /* ======================================================
       BUTTONS
       ====================================================== */
    .stButton > button {
        border-radius: 10px;
        font-weight: 650;
        min-height: 2.7rem;
    }
    /* ======================================================
       INPUTS
       ====================================================== */
    textarea,
    input {
        border-radius: 10px !important;
    }
    </style>
    """,
    unsafe_allow_html=True
)


# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:

    st.markdown("## 🛡️ NewsGuard AI")

    st.caption(
        "AI-powered fake news detection "
        "using Natural Language Processing."
    )

    st.divider()

    st.markdown("### 🤖 Model")

    st.write("**Algorithm:** Linear SVM")
    st.write("**Features:** TF-IDF")
    st.write("**N-grams:** 1–2")
    st.write("**Maximum Features:** 100,000")

    st.divider()

    st.markdown("### 📊 Model Performance")

    col1, col2 = st.columns(2)

    with col1:

        st.metric(
            "Accuracy",
            "96.65%"
        )

    with col2:

        st.metric(
            "ROC-AUC",
            "99.48%"
        )

    st.divider()

    st.markdown("### 🧠 How it works")

    st.caption(
        "1️⃣ Text is combined\n\n"
        "2️⃣ TF-IDF extracts linguistic features\n\n"
        "3️⃣ Linear SVM analyzes the features\n\n"
        "4️⃣ The model produces a classification"
    )

    st.divider()

    st.markdown("### ℹ️ About")

    st.caption(
        "NewsGuard AI is an educational machine-learning "
        "project designed to identify patterns associated "
        "with real and fake news."
    )

    st.caption(
        "The model does not independently verify facts."
    )


# ============================================================
# HERO
# ============================================================

st.markdown(
    """
    <div class="hero">
        <div class="hero-badge">
            🧠 MACHINE LEARNING • NLP
        </div>
        <div class="hero-title">
            🛡️ NewsGuard AI
        </div>
        <div class="hero-subtitle">
            AI-Powered Fake News Detection
        </div>
        <div class="hero-description">
            Analyze news articles using Natural Language Processing
            and a trained Linear Support Vector Machine. NewsGuard AI
            identifies patterns in text associated with real and fake
            news.
        </div>
    </div>
    """,
    unsafe_allow_html=True
)


# ============================================================
# ANALYZER
# ============================================================

st.markdown(
    '<div class="section-title">📰 Analyze News Article</div>',
    unsafe_allow_html=True
)

st.markdown(
    """
    <div class="section-description">
        Enter the headline and article text below.
        The model will analyze the combined content.
    </div>
    """,
    unsafe_allow_html=True
)


# ============================================================
# INPUT
# ============================================================

title = st.text_input(
    "News Headline",
    placeholder="Example: Government announces new education policy",
    key="title_input"
)


article = st.text_area(
    "News Article",
    placeholder="Paste the complete news article here...",
    height=280,
    key="article_input"
)


# ============================================================
# BUTTONS
# ============================================================

col1, col2, col3 = st.columns([3, 1, 1])


with col1:

    analyze = st.button(
        "🔍 Analyze Article",
        type="primary",
        use_container_width=True
    )


with col2:

    st.button(
        "💡 Use Sample",
        use_container_width=True,
        on_click=load_sample
    )


with col3:

    st.button(
        "🗑️ Clear",
        use_container_width=True,
        on_click=clear_inputs
    )


# ============================================================
# PREDICTION
# ============================================================

if analyze:

    if not article.strip():

        st.warning(
            "⚠️ Please enter a news article first."
        )

    else:

        try:

            with st.spinner(
                "🤖 NewsGuard AI is analyzing the article..."
            ):

                result = predict_news(
                    title,
                    article
                )

            st.session_state.analysis_result = result

        except Exception as error:

            st.error("❌ Prediction error")

            st.exception(error)


# ============================================================
# DISPLAY RESULT
# ============================================================

result = st.session_state.analysis_result


if result is not None:

    st.divider()

    st.markdown(
        '<div class="section-title">📊 Analysis Result</div>',
        unsafe_allow_html=True
    )

    prediction = result["prediction"]


    # ========================================================
    # RESULT CARD
    # ========================================================

    if prediction == "REAL":

        st.markdown(
            """
            <div class="result-real">
                <div class="result-title">
                    🟢 LIKELY REAL
                </div>
                <div class="result-description">
                    The model classified this article as likely
                    real based on patterns learned from the
                    training dataset.
                </div>
            </div>
            """,
            unsafe_allow_html=True
        )

    else:

        st.markdown(
            """
            <div class="result-fake">
                <div class="result-title">
                    🔴 LIKELY FAKE
                </div>
                <div class="result-description">
                    The model classified this article as likely
                    fake based on patterns learned from the
                    training dataset.
                </div>
            </div>
            """,
            unsafe_allow_html=True
        )


    # ========================================================
    # MODEL SIGNAL
    # ========================================================

    st.markdown(
        '<div class="section-title">📈 Model Signal</div>',
        unsafe_allow_html=True
    )


    col1, col2 = st.columns(2)


    with col1:

        st.markdown(
            f"""
            <div class="metric-card">
                <div class="metric-label">
                    Decision Score
                </div>
                <div class="metric-value">
                    {result["decision_score"]:.4f}
                </div>
            </div>
            """,
            unsafe_allow_html=True
        )


    with col2:

        st.markdown(
            f"""
            <div class="metric-card">
                <div class="metric-label">
                    Signal Strength
                </div>
                <div class="metric-value">
                    {result["signal_strength"]}
                </div>
            </div>
            """,
            unsafe_allow_html=True
        )


    # ========================================================
    # SIGNAL VISUALIZATION
    # ========================================================

    signal = result["signal_strength"]


    if signal == "LOW":

        fill_class = "signal-fill-low"

        signal_description = (
            "Weak model signal — interpret the prediction carefully."
        )

    elif signal == "MEDIUM":

        fill_class = "signal-fill-medium"

        signal_description = (
            "Moderate model signal — the classification has "
            "some supporting evidence."
        )

    else:

        fill_class = "signal-fill-high"

        signal_description = (
            "Strong model signal — the text strongly matches "
            "patterns learned by the classifier."
        )


    st.markdown(
        f"""
        <div class="signal-container">
            <div class="signal-label">
                <span>Signal strength</span>
                <span>{signal}</span>
            </div>
            <div class="signal-track">
                <div class="{fill_class}"></div>
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )


    # ========================================================
    # SIGNAL EXPLANATION
    # ========================================================

    st.info(
        "💡 " + result["signal_explanation"]
    )


    # ========================================================
    # ARTICLE STATISTICS
    # ========================================================

    st.markdown(
        '<div class="section-title">📄 Article Statistics</div>',
        unsafe_allow_html=True
    )


    col1, col2, col3 = st.columns(3)


    with col1:

        st.markdown(
            f"""
            <div class="metric-card">
                <div class="metric-label">
                    Words
                </div>
                <div class="metric-value">
                    {result["article_words"]}
                </div>
            </div>
            """,
            unsafe_allow_html=True
        )


    with col2:

        st.markdown(
            f"""
            <div class="metric-card">
                <div class="metric-label">
                    Characters
                </div>
                <div class="metric-value">
                    {result["article_characters"]}
                </div>
            </div>
            """,
            unsafe_allow_html=True
        )


    with col3:

        st.markdown(
            f"""
            <div class="metric-card">
                <div class="metric-label">
                    Title Characters
                </div>
                <div class="metric-value">
                    {result["title_characters"]}
                </div>
            </div>
            """,
            unsafe_allow_html=True
        )


    # ========================================================
    # WARNINGS
    # ========================================================

    if result["warning"]:

        st.warning(
            "⚠️ " + result["warning"]
        )


    # ========================================================
    # RESPONSIBLE AI
    # ========================================================

    with st.expander("🛡️ Responsible AI & Limitations"):

        st.write(
            """
            NewsGuard AI is an educational machine-learning
            classification system.

            The model identifies patterns learned from its
            training dataset. It does **not** independently
            verify claims, search the internet, or establish
            whether an article is factually true or false.

            Therefore, predictions should be treated as an
            analytical signal rather than a definitive
            fact-check.
            """
        )


# ============================================================
# ABOUT
# ============================================================

st.divider()

st.markdown(
    '<div class="section-title">ℹ️ About NewsGuard AI</div>',
    unsafe_allow_html=True
)

st.write(
    "NewsGuard AI combines Natural Language Processing "
    "with supervised machine learning to analyze news text."
)


col1, col2, col3 = st.columns(3)


with col1:

    st.markdown(
        """
        <div class="about-card">
            <div class="about-icon">
                🧠
            </div>
            <div class="about-title">
                Natural Language Processing
            </div>
            <div class="about-text">
                News articles are treated as text data and
                transformed into numerical representations
                that a machine-learning model can understand.
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )


with col2:

    st.markdown(
        """
        <div class="about-card">
            <div class="about-icon">
                🔤
            </div>
            <div class="about-title">
                TF-IDF
            </div>
            <div class="about-text">
                TF-IDF converts words and phrases into numerical
                features based on their importance within the
                news text.
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )


with col3:

    st.markdown(
        """
        <div class="about-card">
            <div class="about-icon">
                ⚡
            </div>
            <div class="about-title">
                Linear SVM
            </div>
            <div class="about-text">
                A Linear Support Vector Machine uses the extracted
                features to classify the article into the learned
                REAL or FAKE categories.
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )


# ============================================================
# FOOTER
# ============================================================

st.markdown(
    """
    <div class="footer">
        🛡️ <b>NewsGuard AI</b><br>
        Built with Python • Scikit-learn • Streamlit
    </div>
    """,
    unsafe_allow_html=True
)