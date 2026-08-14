import pandas as pd
import matplotlib.pyplot as plt


# --------------------------------------------------
# Load cleaned dataset
# --------------------------------------------------

df = pd.read_csv(
    "data/WELFake_cleaned.csv",
    keep_default_na=False
)

print("\n========== DATASET ==========")
print(f"Rows: {len(df)}")
print(f"Columns: {len(df.columns)}")


# --------------------------------------------------
# Create text length features
# --------------------------------------------------

df["title_length"] = df["title"].str.len()

df["text_length"] = df["text"].str.len()

df["word_count"] = df["text"].str.split().str.len()

# --------------------------------------------------
# Short article analysis
# --------------------------------------------------

print("\n========== VERY SHORT ARTICLES ==========")

short_articles = df[df["word_count"] < 20]

print(
    f"Articles with fewer than 20 words: "
    f"{len(short_articles)}"
)


# --------------------------------------------------
# Extremely long articles
# --------------------------------------------------

print("\n========== VERY LONG ARTICLES ==========")

long_articles = df[df["word_count"] > 5000]

print(
    f"Articles with more than 5000 words: "
    f"{len(long_articles)}"
)


# --------------------------------------------------
# Empty titles
# --------------------------------------------------

print("\n========== EMPTY TITLES ==========")

print(
    f"Articles with empty titles: "
    f"{(df['title'].str.strip() == '').sum()}"
)
# --------------------------------------------------
# Basic statistics
# --------------------------------------------------

print("\n========== TITLE LENGTH ==========")

print(
    df["title_length"].describe()
)


print("\n========== ARTICLE CHARACTER LENGTH ==========")

print(
    df["text_length"].describe()
)


print("\n========== ARTICLE WORD COUNT ==========")

print(
    df["word_count"].describe()
)


# --------------------------------------------------
# Statistics by label
# --------------------------------------------------

print("\n========== AVERAGE LENGTH BY LABEL ==========")

print(
    df.groupby("label")[
        ["title_length", "text_length", "word_count"]
    ].mean().round(2)
)


# --------------------------------------------------
# Label distribution
# --------------------------------------------------

print("\n========== LABEL DISTRIBUTION ==========")

print(
    df["label"].value_counts()
    .sort_index()
)


# --------------------------------------------------
# Plot label distribution
# --------------------------------------------------

df["label"].value_counts().sort_index().plot(
    kind="bar"
)

plt.title("Class Distribution")

plt.xlabel("Label")

plt.ylabel("Number of Articles")

plt.xticks(
    [0, 1],
    ["Class 0", "Class 1"],
    rotation=0
)

plt.tight_layout()

plt.savefig(
    "assets/class_distribution.png",
    dpi=300
)

plt.show()


# --------------------------------------------------
# Plot article word-count distribution
# --------------------------------------------------

plt.figure()

df["word_count"].clip(
    upper=df["word_count"].quantile(0.99)
).hist(
    bins=50
)

plt.title("Article Word Count Distribution")

plt.xlabel("Number of Words")

plt.ylabel("Number of Articles")

plt.tight_layout()

plt.savefig(
    "assets/article_word_count_distribution.png",
    dpi=300
)

plt.show()