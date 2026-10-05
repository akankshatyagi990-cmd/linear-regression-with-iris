import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, Button
from pathlib import Path


# ============================================================
# STEP 5 - UNDERSTAND THE REGRESSION LINE
# ============================================================


# ------------------------------------------------------------
# 1. DEFINE PROJECT PATH
# ------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[1]

DATA_FILE = PROJECT_ROOT / "data" / "iris_encoded.csv"


# ------------------------------------------------------------
# 2. LOAD DATASET
# ------------------------------------------------------------

df = pd.read_csv(DATA_FILE)


print("\n" + "=" * 60)
print("STEP 5 - UNDERSTAND THE REGRESSION LINE")
print("=" * 60)


# ------------------------------------------------------------
# 3. SELECT X AND y
# ------------------------------------------------------------

X = df["sepal.length"].to_numpy()

y = df["petal.length"].to_numpy()


print("\nSelected variables:")

print("X = sepal.length")
print("y = petal.length")


# ------------------------------------------------------------
# 4. REGRESSION LINE IDEA
# ------------------------------------------------------------

print("\n" + "=" * 60)
print("REGRESSION LINE EQUATION")
print("=" * 60)

print(
    "Predicted y = Intercept + (Slope × X)"
)

print()

print(
    "Slope     = how much predicted y changes "
    "when X increases by 1"
)

print(
    "Intercept = predicted y when X = 0"
)


# ------------------------------------------------------------
# 5. START WITH A MANUAL TRIAL LINE
# ------------------------------------------------------------

# IMPORTANT:
#
# These are NOT the correct regression values.
#
# We are choosing them manually so we can understand
# how changing slope and intercept changes the line.

initial_slope = 1.5

initial_intercept = -5.0


# ------------------------------------------------------------
# 6. CREATE X VALUES FOR DRAWING THE LINE
# ------------------------------------------------------------

line_x = np.linspace(
    X.min(),
    X.max(),
    200,
)


# ------------------------------------------------------------
# 7. CALCULATE TRIAL LINE
# ------------------------------------------------------------

line_y = (
    initial_intercept
    +
    initial_slope * line_x
)


# ------------------------------------------------------------
# 8. CREATE GRAPH
# ------------------------------------------------------------

fig, ax = plt.subplots(
    figsize=(10, 7)
)


# Leave space at the bottom for sliders

plt.subplots_adjust(
    bottom=0.28
)


# Plot actual Iris observations

ax.scatter(
    X,
    y,
    alpha=0.7,
    label="Actual Iris Data",
)


# Plot our manually chosen line

regression_line, = ax.plot(
    line_x,
    line_y,
    linewidth=2,
    label="Trial Regression Line",
)


# ------------------------------------------------------------
# 9. LABEL GRAPH
# ------------------------------------------------------------

ax.set_xlabel(
    "Sepal Length (X)"
)

ax.set_ylabel(
    "Petal Length (y)"
)

ax.set_title(
    "Understanding Slope and Intercept"
)

ax.grid(
    alpha=0.3
)

ax.legend()


# ------------------------------------------------------------
# 10. ADD EQUATION TEXT
# ------------------------------------------------------------

equation_text = ax.text(
    0.03,
    0.95,
    (
        f"Predicted y = "
        f"{initial_intercept:.2f} "
        f"+ ({initial_slope:.2f} × X)"
    ),
    transform=ax.transAxes,
    verticalalignment="top",
)


# ------------------------------------------------------------
# 11. CREATE SLIDER AREAS
# ------------------------------------------------------------

slope_axis = plt.axes(
    [0.20, 0.14, 0.65, 0.03]
)

intercept_axis = plt.axes(
    [0.20, 0.08, 0.65, 0.03]
)


# ------------------------------------------------------------
# 12. CREATE SLOPE SLIDER
# ------------------------------------------------------------

slope_slider = Slider(
    ax=slope_axis,
    label="Slope",
    valmin=0.0,
    valmax=3.0,
    valinit=initial_slope,
    valstep=0.05,
)


# ------------------------------------------------------------
# 13. CREATE INTERCEPT SLIDER
# ------------------------------------------------------------

intercept_slider = Slider(
    ax=intercept_axis,
    label="Intercept",
    valmin=-10.0,
    valmax=5.0,
    valinit=initial_intercept,
    valstep=0.1,
)


# ------------------------------------------------------------
# 14. FUNCTION TO UPDATE THE LINE
# ------------------------------------------------------------

def update_line(value):

    current_slope = slope_slider.val

    current_intercept = intercept_slider.val


    new_line_y = (
        current_intercept
        +
        current_slope * line_x
    )


    regression_line.set_ydata(
        new_line_y
    )


    equation_text.set_text(
        (
            f"Predicted y = "
            f"{current_intercept:.2f} "
            f"+ ({current_slope:.2f} × X)"
        )
    )


    fig.canvas.draw_idle()


# ------------------------------------------------------------
# 15. CONNECT SLIDERS
# ------------------------------------------------------------

slope_slider.on_changed(
    update_line
)

intercept_slider.on_changed(
    update_line
)


# ------------------------------------------------------------
# 16. RESET BUTTON
# ------------------------------------------------------------

reset_axis = plt.axes(
    [0.87, 0.03, 0.08, 0.04]
)

reset_button = Button(
    reset_axis,
    "Reset",
)


def reset(event):

    slope_slider.reset()

    intercept_slider.reset()


reset_button.on_clicked(
    reset
)


# ------------------------------------------------------------
# 17. DISPLAY GRAPH
# ------------------------------------------------------------

print("\n" + "=" * 60)
print("PRACTICAL TASK")
print("=" * 60)

print(
    "Move the Slope slider and watch "
    "the angle of the line change."
)

print(
    "Move the Intercept slider and watch "
    "the whole line move up or down."
)

print()

print(
    "Try to position the line so that it passes "
    "through the overall pattern of the data."
)

print()

print(
    "Do NOT worry about finding the perfect line."
)

print(
    "In Step 6, Python will calculate the best "
    "slope and intercept mathematically."
)


plt.show()


# ------------------------------------------------------------
# 18. COMPLETION
# ------------------------------------------------------------

print("\n" + "=" * 60)
print("STEP 5 COMPLETED")
print("=" * 60)