import pandas as pd
import numpy as np
from pathlib import Path


# ============================================================
# STEP 10B - TEST MODEL WITH OUR OWN VALUE
# ============================================================


# ------------------------------------------------------------
# 1. DEFINE PROJECT PATH
# ------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[1]

DATA_FILE = (
    PROJECT_ROOT
    / "data"
    / "iris_encoded.csv"
)


# ------------------------------------------------------------
# 2. LOAD DATASET
# ------------------------------------------------------------

df = pd.read_csv(
    DATA_FILE
)


print("\n" + "=" * 60)
print("TEST LINEAR REGRESSION WITH YOUR OWN VALUE")
print("=" * 60)


# ------------------------------------------------------------
# 3. SELECT X AND y
# ------------------------------------------------------------

X = df[
    "sepal.length"
].to_numpy()

y = df[
    "petal.length"
].to_numpy()


# ------------------------------------------------------------
# 4. RECREATE THE SAME TRAIN / TEST SPLIT
# ------------------------------------------------------------

total_rows = len(df)

train_ratio = 0.80

random_seed = 42


np.random.seed(
    random_seed
)


indices = np.arange(
    total_rows
)


np.random.shuffle(
    indices
)


train_size = int(
    total_rows
    *
    train_ratio
)


train_indices = indices[
    :train_size
]


X_train = X[
    train_indices
]

y_train = y[
    train_indices
]


# ------------------------------------------------------------
# 5. CALCULATE TRAINING MEANS
# ------------------------------------------------------------

x_train_mean = np.mean(
    X_train
)

y_train_mean = np.mean(
    y_train
)


# ------------------------------------------------------------
# 6. CALCULATE SLOPE
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
# 7. CALCULATE INTERCEPT
# ------------------------------------------------------------

intercept = (
    y_train_mean
    -
    slope
    *
    x_train_mean
)


# ------------------------------------------------------------
# 8. DISPLAY TRAINED MODEL
# ------------------------------------------------------------

print("\nTRAINED MODEL")
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
# 9. SHOW TRAINING DATA RANGE
# ------------------------------------------------------------

minimum_x = np.min(
    X_train
)

maximum_x = np.max(
    X_train
)


print("\nTRAINING SEPAL LENGTH RANGE")
print("=" * 60)


print(
    f"Minimum = {minimum_x:.2f}"
)

print(
    f"Maximum = {maximum_x:.2f}"
)


# ------------------------------------------------------------
# 10. ASK USER FOR OWN VALUE
# ------------------------------------------------------------

print("\n" + "=" * 60)
print("ENTER YOUR OWN VALUE")
print("=" * 60)


while True:

    try:

        user_input = float(
            input(
                "\nEnter Sepal Length: "
            )
        )

        break

    except ValueError:

        print(
            "Please enter a valid number."
        )


# ------------------------------------------------------------
# 11. CHECK WHETHER VALUE IS INSIDE TRAINING RANGE
# ------------------------------------------------------------

if (
    user_input < minimum_x
    or
    user_input > maximum_x
):

    print(
        "\nWARNING:"
    )

    print(
        "Your value is outside the range "
        "seen in the training data."
    )

    print(
        "This is called extrapolation, so the "
        "prediction may be less reliable."
    )

else:

    print(
        "\nYour value is within the range "
        "seen during training."
    )


# ------------------------------------------------------------
# 12. MAKE PREDICTION
# ------------------------------------------------------------

prediction = (
    intercept
    +
    slope
    *
    user_input
)


# ------------------------------------------------------------
# 13. SHOW CALCULATION STEP BY STEP
# ------------------------------------------------------------

print("\n" + "=" * 60)
print("PREDICTION CALCULATION")
print("=" * 60)


print(
    "Formula:"
)

print(
    "Predicted y = Intercept + (Slope × X)"
)


print("\nSubstitute values:")


print(
    f"Predicted y = "
    f"{intercept:.4f} "
    f"+ "
    f"({slope:.4f} × {user_input})"
)


multiplication = (
    slope
    *
    user_input
)


print("\nMultiply first:")


print(
    f"{slope:.4f} × "
    f"{user_input} "
    f"= "
    f"{multiplication:.4f}"
)


print("\nAdd the intercept:")


print(
    f"{intercept:.4f} "
    f"+ "
    f"{multiplication:.4f} "
    f"= "
    f"{prediction:.4f}"
)


# ------------------------------------------------------------
# 14. FINAL RESULT
# ------------------------------------------------------------

print("\n" + "=" * 60)
print("MODEL PREDICTION")
print("=" * 60)


print(
    f"Sepal Length = "
    f"{user_input:.2f}"
)


print(
    f"Predicted Petal Length = "
    f"{prediction:.4f}"
)


# ------------------------------------------------------------
# 15. COMPLETION
# ------------------------------------------------------------

print("\n" + "=" * 60)
print("TEST COMPLETED")
print("=" * 60)