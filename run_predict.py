#!/usr/bin/env python3
"""
Convenience Launcher for Digit Prediction CLI
Usage:
    python run_predict.py [path_to_image]
Example:
    python run_predict.py handwritten_digit_recognition/sample_images/photo_paper_digit_7.jpg
"""
import os
import sys

base_dir = os.path.dirname(os.path.abspath(__file__))
sub_dir = os.path.join(base_dir, "handwritten_digit_recognition")
sys.path.insert(0, sub_dir)

from predict import predict_digit_image

if __name__ == "__main__":
    if len(sys.argv) > 1:
        img_path = sys.argv[1]
    else:
        img_path = os.path.join(sub_dir, "sample_images", "photo_paper_digit_7.jpg")
    predict_digit_image(img_path)

