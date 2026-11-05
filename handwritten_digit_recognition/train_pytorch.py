"""
PyTorch Convolutional Neural Network (CNN) Implementation
==========================================================
Alternative Deep Learning implementation using PyTorch for college students
who want to demonstrate CNN architecture and compare with Deep MLP.

Architecture:
- Conv2D (1 -> 32 filters, 3x3 kernel, ReLU)
- MaxPool2D (2x2)
- Conv2D (32 -> 64 filters, 3x3 kernel, ReLU)
- MaxPool2D (2x2)
- Dropout (0.25)
- Flatten -> Linear (64*5*5 -> 128, ReLU)
- Dropout (0.5)
- Linear (128 -> 10, LogSoftmax)
"""

try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
    from torch.utils.data import DataLoader, TensorDataset
    import numpy as np
    import os

    class DigitCNN(nn.Module):
        def __init__(self):
            super(DigitCNN, self).__init__()
            self.features = nn.Sequential(
                nn.Conv2d(1, 32, kernel_size=3, padding=1),
                nn.ReLU(),
                nn.MaxPool2d(2, 2),
                nn.Conv2d(32, 64, kernel_size=3, padding=1),
                nn.ReLU(),
                nn.MaxPool2d(2, 2),
                nn.Dropout(0.25)
            )
            self.classifier = nn.Sequential(
                nn.Flatten(),
                nn.Linear(64 * 7 * 7, 128),
                nn.ReLU(),
                nn.Dropout(0.5),
                nn.Linear(128, 10)
            )

        def forward(self, x):
            x = self.features(x)
            x = self.classifier(x)
            return x

    def train_pytorch_model():
        base_dir = os.path.dirname(__file__)
        data_path = os.path.join(base_dir, "data", "mnist.npz")
        if not os.path.exists(data_path):
            print("MNIST data not found. Please run download_mnist.py first.")
            return

        print("Loading MNIST for PyTorch CNN training...")
        data = np.load(data_path)
        x_train = torch.tensor(data["x_train"], dtype=torch.float32).unsqueeze(1) / 255.0
        y_train = torch.tensor(data["y_train"], dtype=torch.long)
        x_test = torch.tensor(data["x_test"], dtype=torch.float32).unsqueeze(1) / 255.0
        y_test = torch.tensor(data["y_test"], dtype=torch.long)

        train_loader = DataLoader(TensorDataset(x_train, y_train), batch_size=64, shuffle=True)
        test_loader = DataLoader(TensorDataset(x_test, y_test), batch_size=1000, shuffle=False)

        device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
        print(f"Using compute device: {device}")

        model = DigitCNN().to(device)
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=0.001)

        print("\nStarting CNN Training (5 epochs)...")
        for epoch in range(1, 6):
            model.train()
            running_loss = 0.0
            correct = 0
            total = 0
            for images, labels in train_loader:
                images, labels = images.to(device), labels.to(device)
                optimizer.zero_grad()
                outputs = model(images)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()

                running_loss += loss.item() * images.size(0)
                _, predicted = outputs.max(1)
                total += labels.size(0)
                correct += predicted.eq(labels).sum().item()

            train_acc = 100.0 * correct / total
            print(f"Epoch {epoch}/5 | Loss: {running_loss/total:.4f} | Accuracy: {train_acc:.2f}%")

        # Test evaluation
        model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for images, labels in test_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                _, predicted = outputs.max(1)
                total += labels.size(0)
                correct += predicted.eq(labels).sum().item()

        test_acc = 100.0 * correct / total
        print(f"\nPyTorch CNN Test Accuracy: {test_acc:.2f}%")

        torch.save(model.state_dict(), os.path.join(base_dir, "models", "pytorch_digit_cnn.pth"))
        print("Saved PyTorch CNN model to models/pytorch_digit_cnn.pth")

    if __name__ == "__main__":
        train_pytorch_model()

except ImportError:
    print("PyTorch is not installed in the current environment.")
    print("To run this CNN script, install PyTorch with: pip install torch")
    print("Note: The core NumPy Deep Neural Network (neural_network.py) is fully functional and already trained!")

