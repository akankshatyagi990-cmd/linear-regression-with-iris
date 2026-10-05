import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path


# ============================================================
# STEP 4 - VISUALIZE X VS y
# ============================================================


# ------------------------------------------------------------
# 1. DEFINE PROJECT PATHS
# ------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[1]

DATA_FILE = PROJECT_ROOT / "data" / "iris_encoded.csv"

PLOTS_FOLDER = PROJECT_ROOT / "plots"

OUTPUT_PLOT = PLOTS_FOLDER / "step4_sepal_vs_petal.png"


# ------------------------------------------------------------
# 2. CREATE PLOTS FOLDER IF IT DOES NOT EXIST
# ------------------------------------------------------------

PLOTS_FOLDER.mkdir(
    exist_ok=True
)


# ------------------------------------------------------------
# 3. LOAD THE DATASET
# ------------------------------------------------------------

df = pd.read_csv(DATA_FILE)


print("\n" + "=" * 60)
print("STEP 4 - VISUALIZE X VS y")
print("=" * 60)


# ------------------------------------------------------------
# 4. SELECT X AND y
# ------------------------------------------------------------

X = df["sepal.length"]

y = df["petal.length"]


print("\nSelected variables:")

print("X = sepal.length")
print("y = petal.length")


# ------------------------------------------------------------
# 5. CHECK THE FIRST FEW X-y PAIRS
# ------------------------------------------------------------

xy_data = pd.DataFrame(
    {
        "Sepal Length (X)": X,
        "Petal Length (y)": y,
    }
)


print("\nFIRST 10 X-y PAIRS")
print("=" * 60)

print(
    xy_data.head(10)
)


# ------------------------------------------------------------
# 6. CREATE THE SCATTER PLOT
# ------------------------------------------------------------

plt.figure(
    figsize=(10, 6)
)


plt.scatter(
    X,
    y,
    alpha=0.7,
)


# ------------------------------------------------------------
# 7. ADD LABELS
# ------------------------------------------------------------

plt.xlabel(
    "Sepal Length"
)

plt.ylabel(
    "Petal Length"
)

plt.title(
    "Sepal Length vs Petal Length"
)


# ------------------------------------------------------------
# 8. ADD GRID
# ------------------------------------------------------------

plt.grid(
    alpha=0.3
)


# ------------------------------------------------------------
# 9. SAVE THE GRAPH
# ------------------------------------------------------------

plt.savefig(
    OUTPUT_PLOT,
    dpi=300,
    bbox_inches="tight",
)


print("\n" + "=" * 60)
print("PLOT CREATED")
print("=" * 60)

print(
    f"Plot saved to:\n{OUTPUT_PLOT}"
)


# ------------------------------------------------------------
# 10. DISPLAY THE GRAPH
# ------------------------------------------------------------

plt.show()


# ------------------------------------------------------------
# 11. COMPLETION
# ------------------------------------------------------------

print("\n" + "=" * 60)
print("STEP 4 COMPLETED SUCCESSFULLY")
print("=" * 60)