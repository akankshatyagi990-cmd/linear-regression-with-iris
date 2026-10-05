import pandas as pd
from pathlib import Path


# ============================================================
# STEP 3 - CHOOSE X AND y
# ============================================================


# ------------------------------------------------------------
# 1. DEFINE PROJECT PATH
# ------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[1]

DATA_FILE = PROJECT_ROOT / "data" / "iris_encoded.csv"


# ------------------------------------------------------------
# 2. LOAD THE ENCODED DATASET
# ------------------------------------------------------------

df = pd.read_csv(DATA_FILE)


print("\n" + "=" * 60)
print("STEP 3 - CHOOSE X AND y")
print("=" * 60)


# ------------------------------------------------------------
# 3. VIEW AVAILABLE COLUMNS
# ------------------------------------------------------------

print("\nAVAILABLE COLUMNS")
print("=" * 60)

for column in df.columns:
    print(column)


# ------------------------------------------------------------
# 4. CHOOSE INPUT FEATURE X
# ------------------------------------------------------------

X = df["sepal.length"]


# ------------------------------------------------------------
# 5. CHOOSE TARGET y
# ------------------------------------------------------------

y = df["petal.length"]


# ------------------------------------------------------------
# 6. EXPLAIN OUR CHOICE
# ------------------------------------------------------------

print("\nSELECTED VARIABLES")
print("=" * 60)

print("X = sepal.length")
print("y = petal.length")


print("\nMeaning:")

print(
    "We will use sepal length as the input feature."
)

print(
    "We will try to predict petal length."
)


# ------------------------------------------------------------
# 7. VIEW FIRST 10 X VALUES
# ------------------------------------------------------------

print("\nFIRST 10 X VALUES")
print("=" * 60)

print(X.head(10))


# ------------------------------------------------------------
# 8. VIEW FIRST 10 y VALUES
# ------------------------------------------------------------

print("\nFIRST 10 y VALUES")
print("=" * 60)

print(y.head(10))


# ------------------------------------------------------------
# 9. VIEW X AND y TOGETHER
# ------------------------------------------------------------

xy_data = pd.DataFrame(
    {
        "sepal.length (X)": X,
        "petal.length (y)": y,
    }
)


print("\nFIRST 10 X-y PAIRS")
print("=" * 60)

print(
    xy_data.head(10)
)


# ------------------------------------------------------------
# 10. CHECK NUMBER OF OBSERVATIONS
# ------------------------------------------------------------

print("\nNUMBER OF OBSERVATIONS")
print("=" * 60)

print(
    f"X contains {len(X)} values"
)

print(
    f"y contains {len(y)} values"
)


# ------------------------------------------------------------
# 11. CHECK DATA TYPES
# ------------------------------------------------------------

print("\nDATA TYPES")
print("=" * 60)

print(
    f"X data type: {X.dtype}"
)

print(
    f"y data type: {y.dtype}"
)


# ------------------------------------------------------------
# 12. SIMPLE EXAMPLE
# ------------------------------------------------------------

print("\nEXAMPLE")
print("=" * 60)

example_x = X.iloc[0]
example_y = y.iloc[0]

print(
    f"If sepal length is {example_x}, "
    f"the actual petal length is {example_y}."
)


# ------------------------------------------------------------
# 13. COMPLETION
# ------------------------------------------------------------

print("\n" + "=" * 60)
print("STEP 3 COMPLETED SUCCESSFULLY")
print("=" * 60)