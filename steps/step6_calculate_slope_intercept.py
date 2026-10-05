import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path


# ============================================================
# STEP 6 - CALCULATE SLOPE AND INTERCEPT
# ============================================================


# ------------------------------------------------------------
# 1. DEFINE PROJECT PATHS
# ------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[1]

DATA_FILE = PROJECT_ROOT / "data" / "iris_encoded.csv"

PLOTS_FOLDER = PROJECT_ROOT / "plots"

OUTPUT_PLOT = (
    PLOTS_FOLDER
    / "step6_best_fit_regression_line.png"
)


# ------------------------------------------------------------
# 2. CREATE PLOTS FOLDER IF NEEDED
# ------------------------------------------------------------

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
print("STEP 6 - CALCULATE SLOPE AND INTERCEPT")
print("=" * 60)


# ------------------------------------------------------------
# 4. SELECT X AND y
# ------------------------------------------------------------

X = df["sepal.length"].to_numpy()

y = df["petal.length"].to_numpy()


print("\nSelected variables:")

print("X = sepal.length")
print("y = petal.length")


# ------------------------------------------------------------
# 5. CALCULATE THE MEAN OF X AND y
# ------------------------------------------------------------

x_mean = np.mean(X)

y_mean = np.mean(y)


print("\n" + "=" * 60)
print("MEAN VALUES")
print("=" * 60)

print(
    f"Mean of X = {x_mean:.4f}"
)

print(
    f"Mean of y = {y_mean:.4f}"
)


# ------------------------------------------------------------
# 6. CALCULATE DISTANCE FROM THE MEAN
# ------------------------------------------------------------

x_difference = (
    X - x_mean
)

y_difference = (
    y - y_mean
)


print("\nFirst 5 X differences from mean:")

print(
    x_difference[:5]
)


print("\nFirst 5 y differences from mean:")

print(
    y_difference[:5]
)


# ------------------------------------------------------------
# 7. CALCULATE SLOPE NUMERATOR
# ------------------------------------------------------------

# Formula:
#
# numerator =
# Σ((X - X_mean) * (y - y_mean))


numerator = np.sum(
    x_difference
    *
    y_difference
)


print("\n" + "=" * 60)
print("SLOPE CALCULATION")
print("=" * 60)

print(
    f"Numerator = {numerator:.4f}"
)


# ------------------------------------------------------------
# 8. CALCULATE SLOPE DENOMINATOR
# ------------------------------------------------------------

# Formula:
#
# denominator =
# Σ((X - X_mean)^2)


denominator = np.sum(
    x_difference ** 2
)


print(
    f"Denominator = {denominator:.4f}"
)


# ------------------------------------------------------------
# 9. CALCULATE SLOPE
# ------------------------------------------------------------

slope = (
    numerator
    /
    denominator
)


print(
    f"\nSlope = {slope:.4f}"
)


# ------------------------------------------------------------
# 10. CALCULATE INTERCEPT
# ------------------------------------------------------------

# Formula:
#
# intercept =
# y_mean - (slope * x_mean)


intercept = (
    y_mean
    -
    slope * x_mean
)


print(
    f"Intercept = {intercept:.4f}"
)


# ------------------------------------------------------------
# 11. DISPLAY FINAL REGRESSION EQUATION
# ------------------------------------------------------------

print("\n" + "=" * 60)
print("BEST-FIT REGRESSION LINE")
print("=" * 60)


print(
    f"Predicted Petal Length = "
    f"{intercept:.4f} "
    f"+ ({slope:.4f} × Sepal Length)"
)


# ------------------------------------------------------------
# 12. UNDERSTAND THE SLOPE
# ------------------------------------------------------------

print("\n" + "=" * 60)
print("SLOPE INTERPRETATION")
print("=" * 60)


print(
    f"For every 1-unit increase in sepal length,"
)

print(
    f"the predicted petal length increases by "
    f"approximately {slope:.4f} units."
)


# ------------------------------------------------------------
# 13. UNDERSTAND THE INTERCEPT
# ------------------------------------------------------------

print("\n" + "=" * 60)
print("INTERCEPT INTERPRETATION")
print("=" * 60)


print(
    f"If sepal length were 0, the mathematical "
    f"model would predict petal length = "
    f"{intercept:.4f}."
)


print()

print(
    "Important: X = 0 is outside the actual Iris "
    "data range, so the intercept is mainly part "
    "of the mathematical line in this example."
)


# ------------------------------------------------------------
# 14. CALCULATE PREDICTED VALUES FOR DRAWING THE LINE
# ------------------------------------------------------------

predicted_y = (
    intercept
    +
    slope * X
)


# ------------------------------------------------------------
# 15. SORT DATA FOR A CLEAN REGRESSION LINE
# ------------------------------------------------------------

sorted_indices = np.argsort(
    X
)

sorted_X = X[
    sorted_indices
]

sorted_predictions = predicted_y[
    sorted_indices
]


# ------------------------------------------------------------
# 16. CREATE GRAPH
# ------------------------------------------------------------

plt.figure(
    figsize=(10, 6)
)


# Actual observations

plt.scatter(
    X,
    y,
    alpha=0.7,
    label="Actual Iris Data",
)


# Best-fit regression line

plt.plot(
    sorted_X,
    sorted_predictions,
    linewidth=2,
    label="Best-Fit Regression Line",
)


# ------------------------------------------------------------
# 17. LABEL GRAPH
# ------------------------------------------------------------

plt.xlabel(
    "Sepal Length"
)

plt.ylabel(
    "Petal Length"
)

plt.title(
    "Linear Regression Best-Fit Line"
)

plt.grid(
    alpha=0.3
)

plt.legend()


# ------------------------------------------------------------
# 18. DISPLAY EQUATION ON GRAPH
# ------------------------------------------------------------

equation = (
    f"y = {intercept:.2f} "
    f"+ ({slope:.2f} × X)"
)


plt.text(
    0.03,
    0.95,
    equation,
    transform=plt.gca().transAxes,
    verticalalignment="top",
)


# ------------------------------------------------------------
# 19. SAVE GRAPH
# ------------------------------------------------------------

plt.savefig(
    OUTPUT_PLOT,
    dpi=300,
    bbox_inches="tight",
)


print("\n" + "=" * 60)
print("PLOT SAVED")
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
print("STEP 6 COMPLETED SUCCESSFULLY")
print("=" * 60)