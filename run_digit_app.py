#!/usr/bin/env python3
"""
Convenience Launcher for Handwritten Digit Recognition Web App
Run: python run_digit_app.py
"""
import os
import sys

base_dir = os.path.dirname(os.path.abspath(__file__))
sub_dir = os.path.join(base_dir, "handwritten_digit_recognition")
sys.path.insert(0, sub_dir)

from app import run_server

if __name__ == "__main__":
    port = int(sys.argv[1]) if len(sys.argv) > 1 else 8501
    run_server(port)

