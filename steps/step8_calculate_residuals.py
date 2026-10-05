import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path


# ============================================================
# STEP 8 - CALCULATE RESIDUALS
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

PLOTS_FOLDER = (
    PROJECT_ROOT
    / "plots"
)

OUTPUT_FILE = (
    OUTPUT_FOLDER
    / "step8_residuals.csv"
)

OUTPUT_PLOT = (
    PLOTS_FOLDER
    / "step8_residual_plot.png"
)


# ------------------------------------------------------------
# 2. CREATE FOLDERS IF NEEDED
# ------------------------------------------------------------

OUTPUT_FOLDER.mkdir(
    exist_ok=True
)

PLOTS_FOLDER.mkdir(
    exist_ok=True
)


# ------------------------------------------------------------
# 3. LOAD DATASET
# ------------------------------------------------------------

df = pd.read_csv(
    DATA_FILE
)


print("\n" + "=" * 60)
print("STEP 8 - CALCULATE RESIDUALS")
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
# 5. CALCULATE MEANS
# ------------------------------------------------------------

x_mean = np.mean(X)

y_mean = np.mean(y)


# ------------------------------------------------------------
# 6. CALCULATE SLOPE
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


print("\n" + "=" * 60)
print("REGRESSION EQUATION")
print("=" * 60)

print(
    f"Predicted Petal Length = "
    f"{intercept:.4f} "
    f"+ ({slope:.4f} × Sepal Length)"
)


# ------------------------------------------------------------
# 8. MAKE PREDICTIONS FOR ALL FLOWERS
# ------------------------------------------------------------

predicted_y = (
    intercept
    +
    slope * X
)


# ------------------------------------------------------------
# 9. CALCULATE RESIDUALS
# ------------------------------------------------------------

# Residual:
#
# actual value - predicted value


residuals = (
    y
    -
    predicted_y
)


# ------------------------------------------------------------
# 10. CREATE RESULTS TABLE
# ------------------------------------------------------------

results = pd.DataFrame(
    {
        "sepal_length":
            X,

        "actual_petal_length":
            y,

        "predicted_petal_length":
            predicted_y,

        "residual":
            residuals,
    }
)


# ------------------------------------------------------------
# 11. ROUND DISPLAY VALUES
# ------------------------------------------------------------

display_results = (
    results
    .head(10)
    .round(4)
)


print("\n" + "=" * 60)
print("FIRST 10 ACTUAL VS PREDICTED VALUES")
print("=" * 60)

print(
    display_results
)


# ------------------------------------------------------------
# 12. EXPLAIN ONE RESIDUAL MANUALLY
# ------------------------------------------------------------

example_actual = y[0]

example_predicted = predicted_y[0]

example_residual = residuals[0]


print("\n" + "=" * 60)
print("ONE RESIDUAL EXAMPLE")
print("=" * 60)


print(
    f"Sepal Length = {X[0]:.2f}"
)

print(
    f"Actual Petal Length = "
    f"{example_actual:.4f}"
)

print(
    f"Predicted Petal Length = "
    f"{example_predicted:.4f}"
)


print("\nResidual formula:")

print(
    "Residual = Actual - Predicted"
)


print("\nSubstitute values:")

print(
    f"Residual = "
    f"{example_actual:.4f} "
    f"- "
    f"{example_predicted:.4f}"
)


print(
    f"Residual = "
    f"{example_residual:.4f}"
)


# ------------------------------------------------------------
# 13. INTERPRET POSITIVE / NEGATIVE RESIDUAL
# ------------------------------------------------------------

print("\n" + "=" * 60)
print("HOW TO READ A RESIDUAL")
print("=" * 60)


print(
    "Residual = 0"
)

print(
    "Prediction is exactly equal to the actual value."
)

print()


print(
    "Residual > 0"
)

print(
    "Actual value is higher than predicted."
)

print(
    "The model UNDER-PREDICTED."
)

print()


print(
    "Residual < 0"
)

print(
    "Actual value is lower than predicted."
)

print(
    "The model OVER-PREDICTED."
)


# ------------------------------------------------------------
# 14. FIND LARGEST POSITIVE RESIDUAL
# ------------------------------------------------------------

largest_positive_index = np.argmax(
    residuals
)


print("\n" + "=" * 60)
print("LARGEST POSITIVE RESIDUAL")
print("=" * 60)


print(
    f"Sepal Length: "
    f"{X[largest_positive_index]:.2f}"
)

print(
    f"Actual Petal Length: "
    f"{y[largest_positive_index]:.2f}"
)

print(
    f"Predicted Petal Length: "
    f"{predicted_y[largest_positive_index]:.2f}"
)

print(
    f"Residual: "
    f"{residuals[largest_positive_index]:.4f}"
)


# ------------------------------------------------------------
# 15. FIND LARGEST NEGATIVE RESIDUAL
# ------------------------------------------------------------

largest_negative_index = np.argmin(
    residuals
)


print("\n" + "=" * 60)
print("LARGEST NEGATIVE RESIDUAL")
print("=" * 60)


print(
    f"Sepal Length: "
    f"{X[largest_negative_index]:.2f}"
)

print(
    f"Actual Petal Length: "
    f"{y[largest_negative_index]:.2f}"
)

print(
    f"Predicted Petal Length: "
    f"{predicted_y[largest_negative_index]:.2f}"
)

print(
    f"Residual: "
    f"{residuals[largest_negative_index]:.4f}"
)


# ------------------------------------------------------------
# 16. CHECK AVERAGE RESIDUAL
# ------------------------------------------------------------

average_residual = np.mean(
    residuals
)


print("\n" + "=" * 60)
print("AVERAGE RESIDUAL")
print("=" * 60)


print(
    f"Average Residual = "
    f"{average_residual:.10f}"
)


print()

print(
    "For an ordinary least-squares regression line "
    "with an intercept, the residuals usually balance "
    "around zero."
)


# ------------------------------------------------------------
# 17. SAVE RESIDUAL DATA
# ------------------------------------------------------------

results.to_csv(
    OUTPUT_FILE,
    index=False,
)


print("\n" + "=" * 60)
print("RESIDUAL DATA SAVED")
print("=" * 60)

print(
    OUTPUT_FILE
)


# ------------------------------------------------------------
# 18. CREATE RESIDUAL PLOT
# ------------------------------------------------------------

plt.figure(
    figsize=(10, 6)
)


plt.scatter(
    predicted_y,
    residuals,
    alpha=0.7,
)


# Horizontal line at zero residual

plt.axhline(
    y=0,
    linewidth=2,
)


plt.xlabel(
    "Predicted Petal Length"
)

plt.ylabel(
    "Residual"
)

plt.title(
    "Residual Plot"
)

plt.grid(
    alpha=0.3
)


# ------------------------------------------------------------
# 19. SAVE RESIDUAL PLOT
# ------------------------------------------------------------

plt.savefig(
    OUTPUT_PLOT,
    dpi=300,
    bbox_inches="tight",
)


print("\n" + "=" * 60)
print("RESIDUAL PLOT SAVED")
print("=" * 60)

print(
    OUTPUT_PLOT
)


# ------------------------------------------------------------
# 20. SHOW GRAPH
# ------------------------------------------------------------

plt.show()


# ------------------------------------------------------------
# 21. COMPLETION
# ------------------------------------------------------------

print("\n" + "=" * 60)
print("STEP 8 COMPLETED SUCCESSFULLY")
print("=" * 60)