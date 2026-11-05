import os
import urllib.request
import gzip
import struct
import numpy as np

MNIST_DIR = os.path.join(os.path.dirname(__file__), "data")
os.makedirs(MNIST_DIR, exist_ok=True)

URLS = {
    "train_images": "https://storage.googleapis.com/cvdf-datasets/mnist/train-images-idx3-ubyte.gz",
    "train_labels": "https://storage.googleapis.com/cvdf-datasets/mnist/train-labels-idx1-ubyte.gz",
    "test_images": "https://storage.googleapis.com/cvdf-datasets/mnist/t10k-images-idx3-ubyte.gz",
    "test_labels": "https://storage.googleapis.com/cvdf-datasets/mnist/t10k-labels-idx1-ubyte.gz",
}

def download_file(url, target_path):
    print(f"Downloading {os.path.basename(target_path)} from {url}...")
    headers = {'User-Agent': 'Mozilla/5.0'}
    req = urllib.request.Request(url, headers=headers)
    with urllib.request.urlopen(req, timeout=30) as response, open(target_path, 'wb') as out_file:
        data = response.read()
        out_file.write(data)
    print(f"Saved {os.path.basename(target_path)} ({len(data)} bytes).")

def extract_images(gz_path):
    with gzip.open(gz_path, 'rb') as f:
        magic, num_images, rows, cols = struct.unpack(">IIII", f.read(16))
        buffer = f.read(num_images * rows * cols)
        data = np.frombuffer(buffer, dtype=np.uint8)
        data = data.reshape(num_images, rows, cols)
        return data

def extract_labels(gz_path):
    with gzip.open(gz_path, 'rb') as f:
        magic, num_labels = struct.unpack(">II", f.read(8))
        buffer = f.read(num_labels)
        labels = np.frombuffer(buffer, dtype=np.uint8)
        return labels

def main():
    print("=== Downloading Official MNIST Dataset ===")
    for key, url in URLS.items():
        gz_name = os.path.basename(url)
        gz_path = os.path.join(MNIST_DIR, gz_name)
        if not os.path.exists(gz_path) or os.path.getsize(gz_path) == 0:
            download_file(url, gz_path)
        else:
            print(f"Already exists: {gz_name}")

    print("\nVerifying and parsing MNIST files...")
    train_images = extract_images(os.path.join(MNIST_DIR, "train-images-idx3-ubyte.gz"))
    train_labels = extract_labels(os.path.join(MNIST_DIR, "train-labels-idx1-ubyte.gz"))
    test_images = extract_images(os.path.join(MNIST_DIR, "t10k-images-idx3-ubyte.gz"))
    test_labels = extract_labels(os.path.join(MNIST_DIR, "t10k-labels-idx1-ubyte.gz"))

    print(f"Train images shape: {train_images.shape}, labels: {train_labels.shape}")
    print(f"Test images shape: {test_images.shape}, labels: {test_labels.shape}")

    # Also save a compact npz format for lightning fast loading
    npz_path = os.path.join(MNIST_DIR, "mnist.npz")
    np.savez_compressed(
        npz_path,
        x_train=train_images,
        y_train=train_labels,
        x_test=test_images,
        y_test=test_labels
    )
    print(f"Saved compiled dataset archive: {npz_path} ({os.path.getsize(npz_path) / (1024*1024):.2f} MB)")

    # Also export a CSV sample (first 1000 samples) so the student can inspect it in Excel / College Lab
    csv_path = os.path.join(MNIST_DIR, "mnist_sample_1000.csv")
    print(f"Exporting sample CSV for college submission: {csv_path}...")
    with open(csv_path, 'w') as f:
        # Header: label, pixel0, pixel1, ..., pixel783
        header = ["label"] + [f"pixel{i}" for i in range(784)]
        f.write(",".join(header) + "\n")
        for i in range(min(1000, len(train_images))):
            label = str(train_labels[i])
            pixels = [str(p) for p in train_images[i].flatten()]
            f.write(label + "," + "," + ",".join(pixels) + "\n")
    print("CSV sample exported successfully!")

if __name__ == "__main__":
    main()

