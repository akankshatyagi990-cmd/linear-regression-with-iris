import pandas as pd
import numpy as np
from pathlib import Path


# ============================================================
# STEP 9 - TRAIN / TEST SPLIT
# ============================================================


# ------------------------------------------------------------
# 1. DEFINE PROJECT PATHS
# ------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[1]

DATA_FILE = (
    PROJECT_ROOT
    / "data"
    / "iris_encoded.csv"
)

OUTPUT_FOLDER = (
    PROJECT_ROOT
    / "outputs"
)

TRAIN_OUTPUT_FILE = (
    OUTPUT_FOLDER
    / "step9_train_data.csv"
)

TEST_OUTPUT_FILE = (
    OUTPUT_FOLDER
    / "step9_test_data.csv"
)


# ------------------------------------------------------------
# 2. CREATE OUTPUT FOLDER IF NEEDED
# ------------------------------------------------------------

OUTPUT_FOLDER.mkdir(
    exist_ok=True
)


# ------------------------------------------------------------
# 3. LOAD DATASET
# ------------------------------------------------------------

df = pd.read_csv(
    DATA_FILE
)


print("\n" + "=" * 60)
print("STEP 9 - TRAIN / TEST SPLIT")
print("=" * 60)


# ------------------------------------------------------------
# 4. SELECT X AND y
# ------------------------------------------------------------

X = df[
    "sepal.length"
].to_numpy()

y = df[
    "petal.length"
].to_numpy()


print("\nSelected variables:")

print("X = sepal.length")
print("y = petal.length")


# ------------------------------------------------------------
# 5. CHECK TOTAL NUMBER OF RECORDS
# ------------------------------------------------------------

total_rows = len(df)


print("\n" + "=" * 60)
print("DATASET SIZE")
print("=" * 60)

print(
    f"Total number of rows = {total_rows}"
)


# ------------------------------------------------------------
# 6. CHOOSE TRAIN / TEST PERCENTAGE
# ------------------------------------------------------------

train_ratio = 0.80

test_ratio = 0.20


print("\n" + "=" * 60)
print("SPLIT RATIO")
print("=" * 60)

print(
    "Training data = 80%"
)

print(
    "Testing data = 20%"
)


# ------------------------------------------------------------
# 7. CREATE REPRODUCIBLE RANDOM ORDER
# ------------------------------------------------------------

# We use a fixed seed so every run produces
# the same train/test split.

random_seed = 42

np.random.seed(
    random_seed
)


# ------------------------------------------------------------
# 8. CREATE RANDOM ROW INDICES
# ------------------------------------------------------------

indices = np.arange(
    total_rows
)


np.random.shuffle(
    indices
)


print("\nFirst 10 shuffled row indices:")

print(
    indices[:10]
)


# ------------------------------------------------------------
# 9. CALCULATE SPLIT POSITION
# ------------------------------------------------------------

train_size = int(
    total_rows
    *
    train_ratio
)


test_size = (
    total_rows
    -
    train_size
)


print("\n" + "=" * 60)
print("NUMBER OF RECORDS")
print("=" * 60)

print(
    f"Training rows = {train_size}"
)

print(
    f"Testing rows = {test_size}"
)


# ------------------------------------------------------------
# 10. SPLIT INDICES
# ------------------------------------------------------------

train_indices = indices[
    :train_size
]

test_indices = indices[
    train_size:
]


# ------------------------------------------------------------
# 11. CREATE TRAINING DATA
# ------------------------------------------------------------

X_train = X[
    train_indices
]

y_train = y[
    train_indices
]


# ------------------------------------------------------------
# 12. CREATE TEST DATA
# ------------------------------------------------------------

X_test = X[
    test_indices
]

y_test = y[
    test_indices
]


# ------------------------------------------------------------
# 13. CHECK SHAPES
# ------------------------------------------------------------

print("\n" + "=" * 60)
print("TRAIN / TEST SHAPES")
print("=" * 60)

print(
    f"X_train shape = {X_train.shape}"
)

print(
    f"y_train shape = {y_train.shape}"
)

print(
    f"X_test shape = {X_test.shape}"
)

print(
    f"y_test shape = {y_test.shape}"
)


# ------------------------------------------------------------
# 14. CALCULATE TRAINING MEANS
# ------------------------------------------------------------

x_train_mean = np.mean(
    X_train
)

y_train_mean = np.mean(
    y_train
)


print("\n" + "=" * 60)
print("TRAINING DATA MEANS")
print("=" * 60)

print(
    f"Mean X_train = {x_train_mean:.4f}"
)

print(
    f"Mean y_train = {y_train_mean:.4f}"
)


# ------------------------------------------------------------
# 15. CALCULATE SLOPE USING TRAINING DATA ONLY
# ------------------------------------------------------------

numerator = np.sum(
    (
        X_train
        -
        x_train_mean
    )
    *
    (
        y_train
        -
        y_train_mean
    )
)


denominator = np.sum(
    (
        X_train
        -
        x_train_mean
    )
    ** 2
)


slope = (
    numerator
    /
    denominator
)


# ------------------------------------------------------------
# 16. CALCULATE INTERCEPT USING TRAINING DATA ONLY
# ------------------------------------------------------------

intercept = (
    y_train_mean
    -
    slope
    *
    x_train_mean
)


print("\n" + "=" * 60)
print("MODEL LEARNED FROM TRAINING DATA")
print("=" * 60)

print(
    f"Slope = {slope:.4f}"
)

print(
    f"Intercept = {intercept:.4f}"
)


print("\nRegression equation:")

print(
    f"Predicted Petal Length = "
    f"{intercept:.4f} "
    f"+ ({slope:.4f} × Sepal Length)"
)


# ------------------------------------------------------------
# 17. MAKE PREDICTIONS ON TEST DATA
# ------------------------------------------------------------

y_test_predicted = (
    intercept
    +
    slope
    *
    X_test
)


# ------------------------------------------------------------
# 18. CREATE TEST RESULTS TABLE
# ------------------------------------------------------------

test_results = pd.DataFrame(
    {
        "sepal_length":
            X_test,

        "actual_petal_length":
            y_test,

        "predicted_petal_length":
            y_test_predicted,
    }
)


print("\n" + "=" * 60)
print("FIRST 10 TEST PREDICTIONS")
print("=" * 60)

print(
    test_results
    .head(10)
    .round(4)
)


# ------------------------------------------------------------
# 19. CREATE TRAINING DATAFRAME
# ------------------------------------------------------------

train_results = pd.DataFrame(
    {
        "sepal_length":
            X_train,

        "petal_length":
            y_train,
    }
)


# ------------------------------------------------------------
# 20. SAVE TRAINING DATA
# ------------------------------------------------------------

train_results.to_csv(
    TRAIN_OUTPUT_FILE,
    index=False,
)


# ------------------------------------------------------------
# 21. SAVE TEST DATA
# ------------------------------------------------------------

test_results.to_csv(
    TEST_OUTPUT_FILE,
    index=False,
)


print("\n" + "=" * 60)
print("FILES SAVED")
print("=" * 60)

print(
    f"Training data:\n"
    f"{TRAIN_OUTPUT_FILE}"
)

print()

print(
    f"Test predictions:\n"
    f"{TEST_OUTPUT_FILE}"
)


# ------------------------------------------------------------
# 22. EXPLAIN THE PROCESS
# ------------------------------------------------------------

print("\n" + "=" * 60)
print("WHAT HAPPENED?")
print("=" * 60)

print(
    "1. We randomly split the Iris dataset."
)

print(
    "2. The model learned slope and intercept "
    "using only the training data."
)

print(
    "3. The test data was not used to learn "
    "the regression equation."
)

print(
    "4. We then made predictions for the "
    "unseen test data."
)

print(
    "5. In Step 10, we will measure how good "
    "those test predictions are."
)


# ------------------------------------------------------------
# 23. COMPLETION
# ------------------------------------------------------------

print("\n" + "=" * 60)
print("STEP 9 COMPLETED SUCCESSFULLY")
print("=" * 60)