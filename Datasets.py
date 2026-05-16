from sklearn.datasets import make_classification, make_regression
import pandas as pd

print("Choose task type:")
print("1 - Classification")
print("2 - Regression")

choice = input("Enter number: ")

# ---------- CLASSIFICATION ----------
if choice == "1":

    X, y = make_classification(
        n_samples=500,
        n_features=4,
        n_informative=3,
        n_redundant=1,
        n_classes=4,
        n_clusters_per_class=1,
        class_sep=1.0,
        random_state=42
    )

    df = pd.DataFrame(
        X,
        columns=["f1", "f2", "f3", "f4"]
    )

    # int классы
    df["label"] = y.astype(int)

    filename = "dataset_classification.csv"

    print("\nClassification dataset generated!")

# ---------- REGRESSION ----------
elif choice == "2":

    X, y = make_regression(
        n_samples=500,
        n_features=4,
        n_informative=4,
        noise=10.0,
        random_state=42
    )

    df = pd.DataFrame(
        X,
        columns=["f1", "f2", "f3", "f4"]
    )

    # double значения
    df["target"] = y.astype(float)

    filename = "dataset_regression.csv"

    print("\nRegression dataset generated!")

# ---------- ERROR ----------
else:
    print("Incorrect choice")
    exit()

# сохранение
df.to_csv(filename, index=False)

print(f"Saved to: {filename}")
print("\nFirst 5 rows:\n")

print(df.head())
