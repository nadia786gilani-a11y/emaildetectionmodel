"""
Dataset Exporter: Converts binary MNIST to CSV format
=====================================================
Useful for college lab submissions, Excel inspection, and standard ML tabular projects.
"""
import os
import numpy as np

DATA_DIR = os.path.join(os.path.dirname(__file__), "data")

def export_mnist_to_csv(limit_train=5000, limit_test=1000):
    npz_path = os.path.join(DATA_DIR, "mnist.npz")
    data = np.load(npz_path)

    x_train, y_train = data["x_train"], data["y_train"]
    x_test, y_test = data["x_test"], data["y_test"]

    header = "label," + ",".join([f"pixel{i}" for i in range(784)]) + "\n"

    # Export Train CSV
    train_csv = os.path.join(DATA_DIR, f"mnist_train_{limit_train}.csv")
    print(f"Exporting {limit_train} training samples to {train_csv}...")
    with open(train_csv, "w") as f:
        f.write(header)
        for i in range(min(limit_train, len(x_train))):
            pixels = ",".join(map(str, x_train[i].flatten()))
            f.write(f"{y_train[i]},{pixels}\n")

    # Export Test CSV
    test_csv = os.path.join(DATA_DIR, f"mnist_test_{limit_test}.csv")
    print(f"Exporting {limit_test} test samples to {test_csv}...")
    with open(test_csv, "w") as f:
        f.write(header)
        for i in range(min(limit_test, len(x_test))):
            pixels = ",".join(map(str, x_test[i].flatten()))
            f.write(f"{y_test[i]},{pixels}\n")

    print("CSV Datasets exported successfully!")

if __name__ == "__main__":
    export_mnist_to_csv(5000, 1000)

