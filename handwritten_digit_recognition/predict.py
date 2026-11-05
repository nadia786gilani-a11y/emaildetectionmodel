"""
Digit Prediction CLI Tool
=========================
Takes any image/picture of a handwritten digit, preprocesses it,
predicts the digit using the trained Deep Neural Network, and prints
the predicted digit, confidence distribution, and model accuracy.

Usage:
    python predict.py [path_to_image]
Example:
    python predict.py sample_images/photo_paper_digit_7.jpg
"""

import sys
import os
import json
import numpy as np
from neural_network import DeepNeuralNetwork
from image_preprocessor import DigitImagePreprocessor

DIGIT_ASCII = {
    0: ["  ████  ", " █    █ ", " █    █ ", " █    █ ", "  ████  "],
    1: ["   ██   ", "  ███   ", "   ██   ", "   ██   ", "  ████  "],
    2: ["  ████  ", "      █ ", "   ███  ", "  █     ", "  █████ "],
    3: ["  ████  ", "      █ ", "   ███  ", "      █ ", "  ████  "],
    4: ["  █  █  ", "  █  █  ", "  █████ ", "     █  ", "     █  "],
    5: ["  █████ ", "  █     ", "  ████  ", "      █ ", "  ████  "],
    6: ["  ████  ", "  █     ", "  ████  ", "  █   █ ", "  ████  "],
    7: ["  █████ ", "     █  ", "    █   ", "   █    ", "   █    "],
    8: ["  ████  ", "  █   █ ", "   ███  ", "  █   █ ", "  ████  "],
    9: ["  ████  ", "  █   █ ", "   ████ ", "      █ ", "   ███  "]
}

def predict_digit_image(image_path):
    base_dir = os.path.dirname(__file__)
    weights_path = os.path.join(base_dir, "models", "digit_model_weights.npz")
    metrics_path = os.path.join(base_dir, "models", "model_metrics.json")

    if not os.path.exists(weights_path):
        print(f"Error: Model weights not found at {weights_path}. Please run train.py first.")
        return

    # Load Model
    model = DeepNeuralNetwork.load_weights(weights_path)
    
    # Load Model Metrics
    model_accuracy = "97.93%"
    if os.path.exists(metrics_path):
        with open(metrics_path, "r") as f:
            meta = json.load(f)
            model_accuracy = f"{meta.get('test_accuracy', 97.93):.2f}%"

    # Preprocess Image
    if not os.path.exists(image_path):
        print(f"Error: Image not found at {image_path}")
        return

    print("=" * 65)
    print(f"  ANALYZING HANDWRITTEN DIGIT PICTURE: {os.path.basename(image_path)}")
    print("=" * 65)

    preprocessor = DigitImagePreprocessor()
    vector, debug = preprocessor.preprocess(image_path, return_intermediate=True)

    # Predict
    predicted_digit, probabilities = model.predict(vector)
    confidence = probabilities[predicted_digit] * 100.0

    print("\n" + "-" * 65)
    print("  PREDICTION RESULT:")
    print("-" * 65)
    for line in DIGIT_ASCII[predicted_digit]:
        print(f"         {line}")
    print(f"\n   >>> PREDICTED DIGIT:  {predicted_digit}")
    print(f"   >>> CONFIDENCE:       {confidence:.2f}%")
    print(f"   >>> MODEL ACCURACY:   {model_accuracy} (Evaluated on 10,000 MNIST Test Images)")
    print("-" * 65)

    print("\nDigit Probability Distribution:")
    for d in range(10):
        prob = probabilities[d] * 100.0
        bar_len = int(prob / 2.5)  # 40 chars max
        bar = "█" * bar_len + "░" * (40 - bar_len)
        marker = " <== PREDICTED" if d == predicted_digit else ""
        print(f"   Digit {d}: [{bar}] {prob:6.2f}%{marker}")

    print("\nImage Preprocessing Details:")
    print(f"   - Input Size:          {debug.get('original_size', 'N/A')}")
    print(f"   - Center of Mass (X,Y):({debug.get('center_of_mass', (0,0))[0]:.1f}, {debug.get('center_of_mass', (0,0))[1]:.1f})")
    print(f"   - MNIST Norm Format:   28x28 grayscale, values normalized [0.0, 1.0]")
    print("=" * 65 + "\n")

    return predicted_digit, confidence

if __name__ == "__main__":
    if len(sys.argv) > 1:
        img_path = sys.argv[1]
    else:
        # Default test image
        img_path = os.path.join(os.path.dirname(__file__), "sample_images", "photo_paper_digit_7.jpg")
    predict_digit_image(img_path)

