"""
Deep Neural Network (Multi-Layer Perceptron) for MNIST Handwritten Digit Recognition
=====================================================================================
Implemented with NumPy for maximum performance, transparency, and compatibility.

Architecture:
- Input Layer: 784 neurons (28x28 pixels flattened and normalized to [0, 1])
- Hidden Layer 1: 256 neurons + He (Kaiming) Initialization + ReLU Activation + Dropout
- Hidden Layer 2: 128 neurons + He Initialization + ReLU Activation + Dropout
- Output Layer: 10 neurons + Softmax Activation (Digits 0 through 9)

Loss Function:
- Categorical Cross-Entropy Loss: L = - 1/N * sum( y_true * log(y_pred + eps) )

Optimizer:
- Adam Optimizer (Adaptive Moment Estimation with beta1=0.9, beta2=0.999)
"""

import numpy as np
import json
import os

class DeepNeuralNetwork:
    def __init__(self, input_dim=784, hidden1=256, hidden2=128, output_dim=10):
        self.input_dim = input_dim
        self.hidden1 = hidden1
        self.hidden2 = hidden2
        self.output_dim = output_dim

        # He (Kaiming) Normal Initialization for ReLU networks
        np.random.seed(42)
        self.W1 = np.random.randn(input_dim, hidden1) * np.sqrt(2.0 / input_dim)
        self.b1 = np.zeros((1, hidden1), dtype=np.float32)

        self.W2 = np.random.randn(hidden1, hidden2) * np.sqrt(2.0 / hidden1)
        self.b2 = np.zeros((1, hidden2), dtype=np.float32)

        self.W3 = np.random.randn(hidden2, output_dim) * np.sqrt(2.0 / hidden2)
        self.b3 = np.zeros((1, output_dim), dtype=np.float32)

        # Adam Optimizer Moments
        self.mW1, self.vW1 = np.zeros_like(self.W1), np.zeros_like(self.W1)
        self.mb1, self.vb1 = np.zeros_like(self.b1), np.zeros_like(self.b1)
        self.mW2, self.vW2 = np.zeros_like(self.W2), np.zeros_like(self.W2)
        self.mb2, self.vb2 = np.zeros_like(self.b2), np.zeros_like(self.b2)
        self.mW3, self.vW3 = np.zeros_like(self.W3), np.zeros_like(self.W3)
        self.mb3, self.vb3 = np.zeros_like(self.b3), np.zeros_like(self.b3)
        self.t = 0  # Adam timestep

    def relu(self, Z):
        return np.maximum(0, Z)

    def relu_derivative(self, Z):
        return (Z > 0).astype(np.float32)

    def softmax(self, Z):
        # Subtract max for numerical stability (prevent exp overflow)
        expZ = np.exp(Z - np.max(Z, axis=-1, keepdims=True))
        return expZ / np.sum(expZ, axis=-1, keepdims=True)

    def forward(self, X, training=False, dropout_rate=0.2):
        """
        Forward propagation through layers:
        Layer 1: Z1 = X * W1 + b1, A1 = ReLU(Z1), Dropout(A1)
        Layer 2: Z2 = A1 * W2 + b2, A2 = ReLU(Z2), Dropout(A2)
        Layer 3: Z3 = A2 * W3 + b3, A3 = Softmax(Z3)
        """
        self.Z1 = np.dot(X, self.W1) + self.b1
        self.A1 = self.relu(self.Z1)
        
        if training and dropout_rate > 0:
            self.dropout_mask1 = (np.random.rand(*self.A1.shape) >= dropout_rate) / (1.0 - dropout_rate)
            self.A1 *= self.dropout_mask1
        else:
            self.dropout_mask1 = None

        self.Z2 = np.dot(self.A1, self.W2) + self.b2
        self.A2 = self.relu(self.Z2)
        
        if training and dropout_rate > 0:
            self.dropout_mask2 = (np.random.rand(*self.A2.shape) >= dropout_rate) / (1.0 - dropout_rate)
            self.A2 *= self.dropout_mask2
        else:
            self.dropout_mask2 = None

        self.Z3 = np.dot(self.A2, self.W3) + self.b3
        self.A3 = self.softmax(self.Z3)
        return self.A3

    def compute_loss(self, y_true_onehot, y_pred, l2_reg=1e-4):
        """Categorical Cross-Entropy Loss with optional L2 regularization."""
        eps = 1e-12
        y_pred = np.clip(y_pred, eps, 1.0 - eps)
        N = y_true_onehot.shape[0]
        cross_entropy = -np.sum(y_true_onehot * np.log(y_pred)) / N
        l2_loss = 0.5 * l2_reg * (np.sum(self.W1 ** 2) + np.sum(self.W2 ** 2) + np.sum(self.W3 ** 2))
        return float(cross_entropy + l2_loss)

    def backward(self, X, y_true_onehot, learning_rate=0.001, l2_reg=1e-4, beta1=0.9, beta2=0.999):
        """
        Backpropagation and parameter update using Adam Optimizer.
        """
        N = X.shape[0]

        # Output Layer Gradients (Softmax + Cross Entropy simplifies to: A3 - Y)
        dZ3 = (self.A3 - y_true_onehot) / N
        dW3 = np.dot(self.A2.T, dZ3) + l2_reg * self.W3
        db3 = np.sum(dZ3, axis=0, keepdims=True)

        # Layer 2 Gradients
        dA2 = np.dot(dZ3, self.W3.T)
        if self.dropout_mask2 is not None:
            dA2 *= self.dropout_mask2
        dZ2 = dA2 * self.relu_derivative(self.Z2)
        dW2 = np.dot(self.A1.T, dZ2) + l2_reg * self.W2
        db2 = np.sum(dZ2, axis=0, keepdims=True)

        # Layer 1 Gradients
        dA1 = np.dot(dZ2, self.W2.T)
        if self.dropout_mask1 is not None:
            dA1 *= self.dropout_mask1
        dZ1 = dA1 * self.relu_derivative(self.Z1)
        dW1 = np.dot(X.T, dZ1) + l2_reg * self.W1
        db1 = np.sum(dZ1, axis=0, keepdims=True)

        # Adam Optimizer updates
        self.t += 1
        eps = 1e-8

        for param, grad, m, v in [
            (self.W1, dW1, self.mW1, self.vW1),
            (self.b1, db1, self.mb1, self.vb1),
            (self.W2, dW2, self.mW2, self.vW2),
            (self.b2, db2, self.mb2, self.vb2),
            (self.W3, dW3, self.mW3, self.vW3),
            (self.b3, db3, self.mb3, self.vb3),
        ]:
            m[:] = beta1 * m + (1.0 - beta1) * grad
            v[:] = beta2 * v + (1.0 - beta2) * (grad ** 2)
            m_hat = m / (1.0 - beta1 ** self.t)
            v_hat = v / (1.0 - beta2 ** self.t)
            param -= learning_rate * m_hat / (np.sqrt(v_hat) + eps)

    def fit(self, X_train, y_train, X_val, y_val, epochs=15, batch_size=128, lr=0.001, on_epoch_end=None):
        """Trains the deep network using mini-batch Adam optimization."""
        # Convert labels to one-hot vectors
        num_classes = self.output_dim
        N = X_train.shape[0]
        y_train_oh = np.eye(num_classes)[y_train]
        y_val_oh = np.eye(num_classes)[y_val]

        history = {
            "train_loss": [],
            "train_acc": [],
            "val_loss": [],
            "val_acc": []
        }

        print(f"\nStarting training: {epochs} epochs, batch size {batch_size}, lr={lr}")
        print("=" * 65)

        for epoch in range(1, epochs + 1):
            # Shuffle training dataset each epoch
            indices = np.random.permutation(N)
            X_shuffled = X_train[indices]
            y_shuffled = y_train_oh[indices]

            running_loss = 0.0
            correct = 0

            for i in range(0, N, batch_size):
                xb = X_shuffled[i:i+batch_size]
                yb = y_shuffled[i:i+batch_size]

                preds = self.forward(xb, training=True)
                loss = self.compute_loss(yb, preds)
                running_loss += loss * len(xb)
                correct += np.sum(np.argmax(preds, axis=1) == np.argmax(yb, axis=1))

                self.backward(xb, yb, learning_rate=lr)

            train_loss = running_loss / N
            train_acc = (correct / N) * 100.0

            # Validation evaluation
            val_preds = self.forward(X_val, training=False)
            val_loss = self.compute_loss(y_val_oh, val_preds)
            val_acc = (np.sum(np.argmax(val_preds, axis=1) == y_val) / len(y_val)) * 100.0

            history["train_loss"].append(float(train_loss))
            history["train_acc"].append(float(train_acc))
            history["val_loss"].append(float(val_loss))
            history["val_acc"].append(float(val_acc))

            print(f"Epoch {epoch:2d}/{epochs:2d} | Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.2f}% | Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.2f}%")

            if on_epoch_end:
                on_epoch_end(epoch, train_loss, train_acc, val_loss, val_acc)

        return history

    def predict(self, X):
        """
        Predicts digit class and returns confidence probabilities.
        Accepts: 1D array (784,) or 2D array (N, 784).
        """
        X = np.asarray(X, dtype=np.float32)
        single = (X.ndim == 1)
        if single:
            X = X.reshape(1, -1)

        probs = self.forward(X, training=False)
        preds = np.argmax(probs, axis=1)

        if single:
            return int(preds[0]), probs[0]
        return preds, probs

    def evaluate(self, X_test, y_test):
        """
        Evaluates model on test set, computing accuracy, confusion matrix,
        and per-digit precision, recall, and F1-score.
        """
        N = len(y_test)
        preds, probs = self.predict(X_test)
        acc = float(np.sum(preds == y_test) / N) * 100.0

        # Confusion Matrix
        num_classes = self.output_dim
        cm = np.zeros((num_classes, num_classes), dtype=int)
        for t, p in zip(y_test, preds):
            cm[t, p] += 1

        # Per-class metrics
        class_metrics = {}
        for c in range(num_classes):
            tp = cm[c, c]
            fp = np.sum(cm[:, c]) - tp
            fn = np.sum(cm[c, :]) - tp
            precision = float(tp / (tp + fp)) if (tp + fp) > 0 else 0.0
            recall = float(tp / (tp + fn)) if (tp + fn) > 0 else 0.0
            f1 = float(2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0
            class_metrics[str(c)] = {
                "precision": round(precision * 100, 2),
                "recall": round(recall * 100, 2),
                "f1_score": round(f1 * 100, 2),
                "support": int(np.sum(cm[c, :]))
            }

        return {
            "accuracy": round(acc, 2),
            "total_test_samples": N,
            "confusion_matrix": cm.tolist(),
            "class_metrics": class_metrics
        }

    def save_weights(self, filepath):
        """Saves weights and architecture parameters."""
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        np.savez_compressed(
            filepath,
            W1=self.W1, b1=self.b1,
            W2=self.W2, b2=self.b2,
            W3=self.W3, b3=self.b3,
            config=json.dumps({
                "input_dim": self.input_dim,
                "hidden1": self.hidden1,
                "hidden2": self.hidden2,
                "output_dim": self.output_dim
            })
        )
        print(f"Model weights saved successfully to: {filepath}")

    @classmethod
    def load_weights(cls, filepath):
        """Loads trained model weights."""
        data = np.load(filepath, allow_pickle=True)
        config = json.loads(str(data["config"]))
        model = cls(
            input_dim=config["input_dim"],
            hidden1=config["hidden1"],
            hidden2=config["hidden2"],
            output_dim=config["output_dim"]
        )
        model.W1 = data["W1"]
        model.b1 = data["b1"]
        model.W2 = data["W2"]
        model.b2 = data["b2"]
        model.W3 = data["W3"]
        model.b3 = data["b3"]
        print(f"Model loaded from {filepath} (Arch: {config['input_dim']} -> {config['hidden1']} -> {config['hidden2']} -> {config['output_dim']})")
        return model

