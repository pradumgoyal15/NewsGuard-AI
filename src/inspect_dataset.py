import pandas as pd

# Load the dataset
df = pd.read_csv("data/WELFake_Dataset.csv")

# Display the first 5 rows
print("\n========== FIRST 5 ROWS ==========")
print(df.head())

# Display dataset dimensions
print("\n========== DATASET SHAPE ==========")
print(df.shape)

# Display column names
print("\n========== COLUMNS ==========")
print(df.columns.tolist())

# Display information about the dataset
print("\n========== DATASET INFO ==========")
print(df.info())

# Count missing values
print("\n========== MISSING VALUES ==========")
print(df.isnull().sum())

# Count duplicate rows
print("\n========== DUPLICATE ROWS ==========")
print(df.duplicated().sum())

# Display label distribution
print("\n========== LABEL DISTRIBUTION ==========")
print(df["label"].value_counts())