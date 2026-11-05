"""
MNIST Deep Learning Model Training Script
=========================================
Trains the Deep Neural Network on the official MNIST dataset (60,000 training images),
evaluates on 10,000 test images, and exports the trained weights and evaluation metrics.

Author: Nadia (2nd Year B.Tech CSE)
Project: Handwritten Digit Recognition using Deep Learning
"""

import os
import json
import time
import numpy as np
from neural_network import DeepNeuralNetwork

DATA_DIR = os.path.join(os.path.dirname(__file__), "data")
MODELS_DIR = os.path.join(os.path.dirname(__file__), "models")
os.makedirs(MODELS_DIR, exist_ok=True)

def train_digit_classifier(epochs=12, batch_size=128, lr=0.001):
    print("=" * 70)
    print("  HANDWRITTEN DIGIT RECOGNITION - DEEP LEARNING MODEL TRAINING")
    print("  Dataset: MNIST (60,000 Training Samples, 10,000 Test Samples)")
    print("  Student: Nadia, 2nd Year B.Tech CSE")
    print("=" * 70)

    # 1. Load actual MNIST dataset
    npz_path = os.path.join(DATA_DIR, "mnist.npz")
    if not os.path.exists(npz_path):
        raise FileNotFoundError(f"Dataset archive not found at {npz_path}. Run download_mnist.py first.")

    print(f"\n[1/4] Loading MNIST dataset from {npz_path}...")
    data = np.load(npz_path)
    x_train_raw = data["x_train"]
    y_train = data["y_train"]
    x_test_raw = data["x_test"]
    y_test = data["y_test"]

    print(f"      Loaded: {len(x_train_raw)} training images, {len(x_test_raw)} test images.")

    # 2. Preprocess & Normalize
    print("\n[2/4] Preprocessing & Normalizing pixel values [0, 255] -> [0.0, 1.0]...")
    X_train = x_train_raw.reshape(-1, 784).astype(np.float32) / 255.0
    X_test = x_test_raw.reshape(-1, 784).astype(np.float32) / 255.0

    # Split 5,000 samples for validation during training
    val_size = 5000
    X_val = X_train[-val_size:]
    y_val = y_train[-val_size:]
    X_tr = X_train[:-val_size]
    y_tr = y_train[:-val_size]

    # 3. Initialize & Train Deep Neural Network
    print("\n[3/4] Initializing Deep Neural Network Architecture:")
    print("      Input Layer:   784 neurons (28x28 normalized pixels)")
    print("      Hidden Layer 1: 256 neurons (ReLU + He Init + Dropout 0.2)")
    print("      Hidden Layer 2: 128 neurons (ReLU + He Init + Dropout 0.2)")
    print("      Output Layer:   10 neurons (Softmax for Digits 0 - 9)")
    print("      Optimizer:     Adam (lr=0.001, beta1=0.9, beta2=0.999)")
    print("      Loss Function: Categorical Cross-Entropy\n")

    model = DeepNeuralNetwork(input_dim=784, hidden1=256, hidden2=128, output_dim=10)

    start_time = time.time()
    history = model.fit(
        X_train=X_tr,
        y_train=y_tr,
        X_val=X_val,
        y_val=y_val,
        epochs=epochs,
        batch_size=batch_size,
        lr=lr
    )
    duration = time.time() - start_time
    print(f"\nTraining completed in {duration:.2f} seconds ({duration/60:.2f} minutes).")

    # 4. Comprehensive Evaluation on 10,000 Unseen Test Images
    print("\n[4/4] Evaluating Model Accuracy on 10,000 Unseen Test Samples...")
    eval_results = model.evaluate(X_test, y_test)
    test_accuracy = eval_results["accuracy"]

    print("\n" + "=" * 70)
    print(f"  FINAL MODEL TEST ACCURACY: {test_accuracy:.2f}%")
    print("=" * 70)

    print("\nPer-Digit Performance:")
    print(f" {'Digit':<6} | {'Precision (%)':<15} | {'Recall (%)':<15} | {'F1-Score (%)':<15} | {'Support':<8}")
    print("-" * 70)
    for digit in range(10):
        d_str = str(digit)
        m = eval_results["class_metrics"][d_str]
        print(f" {d_str:<6} | {m['precision']:<15.2f} | {m['recall']:<15.2f} | {m['f1_score']:<15.2f} | {m['support']:<8}")
    print("-" * 70)

    # 5. Save Weights & Metrics
    weights_path = os.path.join(MODELS_DIR, "digit_model_weights.npz")
    model.save_weights(weights_path)

    metrics_path = os.path.join(MODELS_DIR, "model_metrics.json")
    save_data = {
        "project": "Handwritten Digit Recognition using Deep Learning",
        "student": "Nadia (2nd Year B.Tech CSE)",
        "dataset": "MNIST",
        "total_training_samples": len(X_train),
        "total_test_samples": len(X_test),
        "test_accuracy": test_accuracy,
        "epochs": epochs,
        "batch_size": batch_size,
        "training_time_seconds": round(duration, 2),
        "history": history,
        "class_metrics": eval_results["class_metrics"],
        "confusion_matrix": eval_results["confusion_matrix"]
    }

    with open(metrics_path, "w") as f:
        json.dump(save_data, f, indent=2)

    print(f"\nEvaluation metrics and training curves saved to: {metrics_path}")
    print("\nModel is ready for live predictions and web app demonstrations!")

if __name__ == "__main__":
    train_digit_classifier(epochs=12, batch_size=128, lr=0.001)

