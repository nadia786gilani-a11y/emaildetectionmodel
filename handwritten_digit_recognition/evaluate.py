"""
Evaluation & Accuracy Report Script
====================================
Evaluates the trained Deep Learning Neural Network on the entire MNIST test dataset
(10,000 samples) and prints confusion matrix, precision, recall, and F1-scores.
"""

import os
import json
import numpy as np
from neural_network import DeepNeuralNetwork

def run_evaluation():
    base_dir = os.path.dirname(__file__)
    data_path = os.path.join(base_dir, "data", "mnist.npz")
    weights_path = os.path.join(base_dir, "models", "digit_model_weights.npz")

    if not os.path.exists(weights_path):
        print(f"Error: Weights file not found at {weights_path}. Run train.py first.")
        return

    print("Loading test data...")
    data = np.load(data_path)
    X_test = data["x_test"].reshape(-1, 784).astype(np.float32) / 255.0
    y_test = data["y_test"]

    print("Loading model...")
    model = DeepNeuralNetwork.load_weights(weights_path)

    print("Running evaluation on 10,000 test samples...")
    results = model.evaluate(X_test, y_test)

    print("\n" + "=" * 70)
    print(f"  MNIST DEEP LEARNING MODEL ACCURACY: {results['accuracy']:.2f}%")
    print(f"  Total Test Samples Evaluated:      {results['total_test_samples']}")
    print("=" * 70)

    print("\nClassification Report (Per-Digit Performance):")
    print(f" {'Digit':<6} | {'Precision (%)':<15} | {'Recall (%)':<15} | {'F1-Score (%)':<15} | {'Support':<8}")
    print("-" * 70)
    for d in range(10):
        m = results["class_metrics"][str(d)]
        print(f" {d:<6} | {m['precision']:<15.2f} | {m['recall']:<15.2f} | {m['f1_score']:<15.2f} | {m['support']:<8}")
    print("-" * 70)

    print("\nConfusion Matrix (Rows: Actual Digit, Columns: Predicted Digit):")
    print("      " + " ".join([f"{d:>5}" for d in range(10)]))
    print("    " + "-" * 56)
    cm = results["confusion_matrix"]
    for row_idx, row in enumerate(cm):
        row_str = " ".join([f"{val:>5}" for val in row])
        print(f"  {row_idx:<2} | {row_str}")
    print("=" * 70 + "\n")

if __name__ == "__main__":
    run_evaluation()

