import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path


# ============================================================
# STEP 10 - EVALUATE MODEL
# MAE, MSE, RMSE AND R²
# ============================================================


# ------------------------------------------------------------
# 1. DEFINE PROJECT PATHS
# ------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[1]

TEST_FILE = (
    PROJECT_ROOT
    / "outputs"
    / "step9_test_data.csv"
)

OUTPUT_FOLDER = (
    PROJECT_ROOT
    / "outputs"
)

PLOTS_FOLDER = (
    PROJECT_ROOT
    / "plots"
)

EVALUATION_FILE = (
    OUTPUT_FOLDER
    / "step10_evaluation_results.csv"
)

METRICS_FILE = (
    OUTPUT_FOLDER
    / "step10_metrics.csv"
)

PLOT_FILE = (
    PLOTS_FOLDER
    / "step10_actual_vs_predicted.png"
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
# 3. LOAD STEP 9 TEST DATA
# ------------------------------------------------------------

df = pd.read_csv(
    TEST_FILE
)


print("\n" + "=" * 60)
print("STEP 10 - MODEL EVALUATION")
print("=" * 60)


print("\nTest data loaded from:")

print(
    TEST_FILE
)


print(
    f"\nNumber of test records = {len(df)}"
)


# ------------------------------------------------------------
# 4. EXTRACT ACTUAL AND PREDICTED VALUES
# ------------------------------------------------------------

actual = df[
    "actual_petal_length"
].to_numpy()

predicted = df[
    "predicted_petal_length"
].to_numpy()


print("\n" + "=" * 60)
print("ACTUAL VS PREDICTED")
print("=" * 60)


print(
    df.head(10).round(4)
)


# ------------------------------------------------------------
# 5. CALCULATE RESIDUALS
# ------------------------------------------------------------

# Residual:
#
# Actual - Predicted


residuals = (
    actual
    -
    predicted
)


# ------------------------------------------------------------
# 6. CALCULATE ABSOLUTE ERRORS
# ------------------------------------------------------------

absolute_errors = np.abs(
    residuals
)


# ------------------------------------------------------------
# 7. CALCULATE SQUARED ERRORS
# ------------------------------------------------------------

squared_errors = (
    residuals ** 2
)


# ------------------------------------------------------------
# 8. ADD ERROR COLUMNS TO DATAFRAME
# ------------------------------------------------------------

df[
    "residual"
] = residuals

df[
    "absolute_error"
] = absolute_errors

df[
    "squared_error"
] = squared_errors


print("\n" + "=" * 60)
print("FIRST 10 ERROR CALCULATIONS")
print("=" * 60)


print(
    df.head(10).round(4)
)


# ============================================================
# MAE
# ============================================================


# ------------------------------------------------------------
# 9. CALCULATE MAE
# ------------------------------------------------------------

# MAE =
# Mean Absolute Error
#
# Average of:
#
# |Actual - Predicted|


mae = np.mean(
    absolute_errors
)


print("\n" + "=" * 60)
print("1. MAE - MEAN ABSOLUTE ERROR")
print("=" * 60)


print(
    f"MAE = {mae:.4f}"
)


print()

print(
    "Meaning:"
)

print(
    "On average, the model's prediction differs "
    f"from the actual petal length by about {mae:.4f} units."
)


# ============================================================
# MSE
# ============================================================


# ------------------------------------------------------------
# 10. CALCULATE MSE
# ------------------------------------------------------------

# MSE =
# Mean Squared Error
#
# Average of:
#
# (Actual - Predicted)^2


mse = np.mean(
    squared_errors
)


print("\n" + "=" * 60)
print("2. MSE - MEAN SQUARED ERROR")
print("=" * 60)


print(
    f"MSE = {mse:.4f}"
)


print()

print(
    "MSE squares the errors, so larger errors "
    "receive a bigger penalty."
)


# ============================================================
# RMSE
# ============================================================


# ------------------------------------------------------------
# 11. CALCULATE RMSE
# ------------------------------------------------------------

# RMSE =
#
# square root of MSE


rmse = np.sqrt(
    mse
)


print("\n" + "=" * 60)
print("3. RMSE - ROOT MEAN SQUARED ERROR")
print("=" * 60)


print(
    f"RMSE = {rmse:.4f}"
)


print()

print(
    "RMSE is expressed in the same unit "
    "as petal length."
)


print(
    "It penalizes larger prediction errors "
    "more strongly than MAE."
)


# ============================================================
# R²
# ============================================================


# ------------------------------------------------------------
# 12. CALCULATE MEAN OF ACTUAL TEST VALUES
# ------------------------------------------------------------

actual_mean = np.mean(
    actual
)


# ------------------------------------------------------------
# 13. CALCULATE SS_RES
# ------------------------------------------------------------

# SS_res =
#
# Sum of squared prediction errors


ss_res = np.sum(
    (
        actual
        -
        predicted
    )
    ** 2
)


# ------------------------------------------------------------
# 14. CALCULATE SS_TOTAL
# ------------------------------------------------------------

# SS_total =
#
# Sum of squared differences between
# each actual value and mean actual value


ss_total = np.sum(
    (
        actual
        -
        actual_mean
    )
    ** 2
)


# ------------------------------------------------------------
# 15. CALCULATE R²
# ------------------------------------------------------------

if ss_total == 0:

    r_squared = 0.0

else:

    r_squared = (
        1
        -
        (
            ss_res
            /
            ss_total
        )
    )


print("\n" + "=" * 60)
print("4. R² - R-SQUARED")
print("=" * 60)


print(
    f"Mean of actual test values = "
    f"{actual_mean:.4f}"
)


print(
    f"SS Residual = "
    f"{ss_res:.4f}"
)


print(
    f"SS Total = "
    f"{ss_total:.4f}"
)


print(
    f"\nR² = "
    f"{r_squared:.4f}"
)


print()

print(
    "R² tells us how much of the variation "
    "in petal length is explained by our "
    "linear relationship with sepal length."
)


# ------------------------------------------------------------
# 16. INTERPRET R²
# ------------------------------------------------------------

print("\nR² interpretation:")


if r_squared >= 0.9:

    print(
        "Very strong explanatory relationship."
    )

elif r_squared >= 0.7:

    print(
        "Strong explanatory relationship."
    )

elif r_squared >= 0.5:

    print(
        "Moderate explanatory relationship."
    )

elif r_squared >= 0:

    print(
        "Weak explanatory relationship."
    )

else:

    print(
        "The model performs worse than simply "
        "predicting the mean of the test target."
    )


print()

print(
    "Important: R² is NOT the same thing "
    "as classification accuracy."
)


# ============================================================
# UNDERSTAND ONE EXAMPLE
# ============================================================


# ------------------------------------------------------------
# 17. INSPECT ONE TEST PREDICTION
# ------------------------------------------------------------

example_actual = actual[0]

example_predicted = predicted[0]

example_residual = residuals[0]

example_absolute_error = absolute_errors[0]

example_squared_error = squared_errors[0]


print("\n" + "=" * 60)
print("ONE PRACTICAL EXAMPLE")
print("=" * 60)


print(
    f"Actual Petal Length = "
    f"{example_actual:.4f}"
)


print(
    f"Predicted Petal Length = "
    f"{example_predicted:.4f}"
)


print(
    f"Residual = "
    f"{example_residual:.4f}"
)


print(
    f"Absolute Error = "
    f"{example_absolute_error:.4f}"
)


print(
    f"Squared Error = "
    f"{example_squared_error:.4f}"
)


# ============================================================
# FINAL METRIC SUMMARY
# ============================================================


print("\n" + "=" * 60)
print("FINAL MODEL EVALUATION")
print("=" * 60)


print(
    f"MAE  = {mae:.4f}"
)

print(
    f"MSE  = {mse:.4f}"
)

print(
    f"RMSE = {rmse:.4f}"
)

print(
    f"R²   = {r_squared:.4f}"
)


# ------------------------------------------------------------
# 18. CREATE METRICS DATAFRAME
# ------------------------------------------------------------

metrics_df = pd.DataFrame(
    {
        "metric": [
            "MAE",
            "MSE",
            "RMSE",
            "R_squared",
        ],

        "value": [
            mae,
            mse,
            rmse,
            r_squared,
        ],
    }
)


# ------------------------------------------------------------
# 19. SAVE DETAILED TEST RESULTS
# ------------------------------------------------------------

df.to_csv(
    EVALUATION_FILE,
    index=False,
)


# ------------------------------------------------------------
# 20. SAVE METRIC SUMMARY
# ------------------------------------------------------------

metrics_df.to_csv(
    METRICS_FILE,
    index=False,
)


print("\n" + "=" * 60)
print("FILES SAVED")
print("=" * 60)


print(
    EVALUATION_FILE
)

print()

print(
    METRICS_FILE
)


# ============================================================
# VISUALIZE ACTUAL VS PREDICTED
# ============================================================


# ------------------------------------------------------------
# 21. CREATE ACTUAL VS PREDICTED SCATTER PLOT
# ------------------------------------------------------------

plt.figure(
    figsize=(9, 6)
)


plt.scatter(
    actual,
    predicted,
    alpha=0.7,
)


# ------------------------------------------------------------
# 22. DRAW PERFECT-PREDICTION LINE
# ------------------------------------------------------------

minimum_value = min(
    actual.min(),
    predicted.min(),
)

maximum_value = max(
    actual.max(),
    predicted.max(),
)


plt.plot(
    [
        minimum_value,
        maximum_value,
    ],
    [
        minimum_value,
        maximum_value,
    ],
    linestyle="--",
    label="Perfect Prediction",
)


# ------------------------------------------------------------
# 23. LABEL GRAPH
# ------------------------------------------------------------

plt.xlabel(
    "Actual Petal Length"
)

plt.ylabel(
    "Predicted Petal Length"
)

plt.title(
    "Actual vs Predicted Petal Length"
)

plt.grid(
    alpha=0.3
)

plt.legend()


# ------------------------------------------------------------
# 24. SAVE GRAPH
# ------------------------------------------------------------

plt.savefig(
    PLOT_FILE,
    dpi=300,
    bbox_inches="tight",
)


print("\nPlot saved to:")

print(
    PLOT_FILE
)


# ------------------------------------------------------------
# 25. SHOW GRAPH
# ------------------------------------------------------------

plt.show()


# ------------------------------------------------------------
# 26. COMPLETION
# ------------------------------------------------------------

print("\n" + "=" * 60)
print("STEP 10 COMPLETED SUCCESSFULLY")
print("=" * 60)