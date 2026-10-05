import pandas as pd
import numpy as np
from pathlib import Path


# ============================================================
# STEP 7 - MAKE PREDICTIONS MANUALLY
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

OUTPUT_FILE = (
    OUTPUT_FOLDER
    / "step7_manual_predictions.csv"
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
print("STEP 7 - MAKE PREDICTIONS MANUALLY")
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

print(
    "X = sepal.length"
)

print(
    "y = petal.length"
)


# ------------------------------------------------------------
# 5. CALCULATE MEAN VALUES
# ------------------------------------------------------------

x_mean = np.mean(X)

y_mean = np.mean(y)


# ------------------------------------------------------------
# 6. CALCULATE SLOPE AGAIN
# ------------------------------------------------------------

numerator = np.sum(
    (X - x_mean)
    *
    (y - y_mean)
)

denominator = np.sum(
    (X - x_mean) ** 2
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
    y_mean
    -
    slope * x_mean
)


# ------------------------------------------------------------
# 8. DISPLAY OUR REGRESSION EQUATION
# ------------------------------------------------------------

print("\n" + "=" * 60)
print("OUR REGRESSION EQUATION")
print("=" * 60)


print(
    f"Predicted Petal Length = "
    f"{intercept:.4f} "
    f"+ ({slope:.4f} × Sepal Length)"
)


# ------------------------------------------------------------
# 9. CREATE OUR OWN PREDICTION FUNCTION
# ------------------------------------------------------------

def predict_petal_length(
    sepal_length
):

    predicted_petal_length = (
        intercept
        +
        slope * sepal_length
    )

    return predicted_petal_length


# ------------------------------------------------------------
# 10. MAKE ONE MANUAL PREDICTION
# ------------------------------------------------------------

manual_sepal_length = 6.0


manual_prediction = (
    predict_petal_length(
        manual_sepal_length
    )
)


print("\n" + "=" * 60)
print("MANUAL PREDICTION EXAMPLE")
print("=" * 60)


print(
    f"Sepal Length = "
    f"{manual_sepal_length}"
)


print("\nUsing:")

print(
    "Predicted y = "
    "Intercept + (Slope × X)"
)


print("\nSubstitute the values:")

print(
    f"Predicted y = "
    f"{intercept:.4f} "
    f"+ "
    f"({slope:.4f} × "
    f"{manual_sepal_length})"
)


multiplication_result = (
    slope
    *
    manual_sepal_length
)


print("\nFirst calculate:")

print(
    f"{slope:.4f} × "
    f"{manual_sepal_length} "
    f"= "
    f"{multiplication_result:.4f}"
)


print("\nThen add the intercept:")

print(
    f"{intercept:.4f} "
    f"+ "
    f"{multiplication_result:.4f}"
)


print(
    f"\nPredicted Petal Length = "
    f"{manual_prediction:.4f}"
)


# ------------------------------------------------------------
# 11. MAKE MULTIPLE MANUAL PREDICTIONS
# ------------------------------------------------------------

new_sepal_lengths = [
    5.0,
    5.5,
    6.0,
    6.5,
    7.0,
]


predicted_petal_lengths = []


for sepal_length in new_sepal_lengths:

    prediction = (
        predict_petal_length(
            sepal_length
        )
    )

    predicted_petal_lengths.append(
        prediction
    )


# ------------------------------------------------------------
# 12. CREATE A RESULTS TABLE
# ------------------------------------------------------------

prediction_results = pd.DataFrame(
    {
        "sepal_length":
            new_sepal_lengths,

        "predicted_petal_length":
            predicted_petal_lengths,
    }
)


print("\n" + "=" * 60)
print("MULTIPLE PREDICTIONS")
print("=" * 60)


print(
    prediction_results.round(4)
)


# ------------------------------------------------------------
# 13. EXPLAIN WHAT IS HAPPENING
# ------------------------------------------------------------

print("\n" + "=" * 60)
print("WHAT THE MODEL IS DOING")
print("=" * 60)


print(
    "For every new Sepal Length:"
)

print()

print(
    "1. Multiply Sepal Length by the slope."
)

print(
    "2. Add the intercept."
)

print(
    "3. The result is the predicted Petal Length."
)


# ------------------------------------------------------------
# 14. SHOW CHANGE BETWEEN TWO PREDICTIONS
# ------------------------------------------------------------

first_x = 6.0

second_x = 7.0


first_prediction = (
    predict_petal_length(
        first_x
    )
)

second_prediction = (
    predict_petal_length(
        second_x
    )
)


difference_in_x = (
    second_x
    -
    first_x
)

difference_in_prediction = (
    second_prediction
    -
    first_prediction
)


print("\n" + "=" * 60)
print("UNDERSTANDING THE SLOPE")
print("=" * 60)


print(
    f"Sepal Length changed from "
    f"{first_x} to {second_x}"
)


print(
    f"Change in X = "
    f"{difference_in_x:.1f}"
)


print(
    f"Predicted Petal Length changed from "
    f"{first_prediction:.4f} "
    f"to "
    f"{second_prediction:.4f}"
)


print(
    f"Change in predicted y = "
    f"{difference_in_prediction:.4f}"
)


print()

print(
    f"This is approximately equal to "
    f"our slope: {slope:.4f}"
)


# ------------------------------------------------------------
# 15. SAVE PREDICTIONS
# ------------------------------------------------------------

prediction_results.to_csv(
    OUTPUT_FILE,
    index=False,
)


print("\n" + "=" * 60)
print("PREDICTIONS SAVED")
print("=" * 60)


print(
    OUTPUT_FILE
)


# ------------------------------------------------------------
# 16. COMPLETION
# ------------------------------------------------------------

print("\n" + "=" * 60)
print("STEP 7 COMPLETED SUCCESSFULLY")
print("=" * 60)