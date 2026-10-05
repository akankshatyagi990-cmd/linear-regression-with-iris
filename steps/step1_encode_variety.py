import pandas as pd
from pathlib import Path


# ============================================================
# STEP 1 - PROJECT FILE PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

INPUT_FILE = PROJECT_ROOT / "data" / "iris.csv"

OUTPUT_FILE = PROJECT_ROOT / "data" / "iris_encoded.csv"


# ============================================================
# STEP 2 - LOAD ORIGINAL DATASET
# ============================================================

df = pd.read_csv(INPUT_FILE)


print("\n" + "=" * 60)
print("ORIGINAL IRIS DATASET")
print("=" * 60)

print(df.head())


# ============================================================
# STEP 3 - CHECK ORIGINAL VARIETY VALUES
# ============================================================

print("\n" + "=" * 60)
print("ORIGINAL VARIETY VALUES")
print("=" * 60)

print(df["variety"].unique())


print("\nOriginal data type:")

print(df["variety"].dtype)


# ============================================================
# STEP 4 - DEFINE ENCODING
# ============================================================

variety_mapping = {
    "Setosa": 0,
    "Versicolor": 1,
    "Virginica": 2,
}


print("\n" + "=" * 60)
print("ENCODING")
print("=" * 60)

print("Setosa      -> 0")
print("Versicolor  -> 1")
print("Virginica   -> 2")


# ============================================================
# STEP 5 - ENCODE VARIETY
# ============================================================

df["variety"] = df["variety"].map(
    variety_mapping
)


# ============================================================
# STEP 6 - VERIFY RESULT
# ============================================================

print("\n" + "=" * 60)
print("DATA AFTER ENCODING")
print("=" * 60)

print(df.head())


print("\nEncoded values:")

print(df["variety"].unique())


print("\nNew data type:")

print(df["variety"].dtype)


# ============================================================
# STEP 7 - CHECK FOR ENCODING PROBLEMS
# ============================================================

missing_values = df["variety"].isnull().sum()


print("\n" + "=" * 60)
print("VALIDATION")
print("=" * 60)

print(
    f"Missing values after encoding: "
    f"{missing_values}"
)


if missing_values == 0:

    print(
        "Encoding successful."
    )

else:

    print(
        "WARNING: Some values were not encoded."
    )


# ============================================================
# STEP 8 - CHECK CATEGORY COUNTS
# ============================================================

print("\n" + "=" * 60)
print("ENCODED CATEGORY COUNTS")
print("=" * 60)

print(
    df["variety"]
    .value_counts()
    .sort_index()
)


# ============================================================
# STEP 9 - SAVE ENCODED DATASET
# ============================================================

df.to_csv(
    OUTPUT_FILE,
    index=False,
)


print("\n" + "=" * 60)
print("FILE CREATED")
print("=" * 60)

print(
    f"Encoded dataset saved as:\n"
    f"{OUTPUT_FILE}"
)


print("\nSTEP 1 COMPLETED SUCCESSFULLY")