# College Project Report: Handwritten Digit Recognition Using Deep Learning

**Student Name:** Nadia  
**Course:** B.Tech Computer Science & Engineering (2nd Year)  
**Subject:** Deep Learning / Machine Learning Mini-Project  
**Dataset:** Modified National Institute of Standards and Technology (MNIST)  
**Model Test Accuracy:** **97.93%** (Evaluated on 10,000 Unseen Test Images)  

---

## 1. Project Overview & Problem Statement
Handwritten digit recognition is a fundamental computer vision problem with widespread applications in bank cheque automation, postal mail sorting, digitized form processing, and automated document indexing.

### Objectives:
1. Train a Deep Neural Network from first principles using the **MNIST dataset** (60,000 training images, 10,000 test images).
2. Take **actual pictures of handwritten digits** (smartphone camera photos, paper scans, or mouse/touch screen drawings).
3. Implement an intelligent **Image Preprocessing Pipeline** that adapts real-world photos (dark ink on light paper, shadows, arbitrary dimensions) to the 28×28 center-of-mass normalized MNIST format.
4. Predict which digit (0 through 9) is written and display the **predicted digit**, **confidence score**, **full probability distribution**, and **overall model test accuracy**.
5. Attach the **actual dataset files** (CSV and NPZ formats) for college evaluation and lab submission.

---

## 2. System Architecture & End-to-End Pipeline

```
  [ Real Picture of Digit ]   (Photo taken from camera / paper / drawn on canvas)
             │
             ▼
  ┌──────────────────────────────────────────────────────────────────┐
  │                   IMAGE PREPROCESSING PIPELINE                   │
  ├──────────────────────────────────────────────────────────────────┤
  │ 1. Grayscale Conversion: (RGB / RGBA -> Single Intensity Channel)│
  │ 2. Border Brightness Detection: Auto-detects light paper        │
  │ 3. Color Inversion: Dark ink on white -> Bright stroke on dark   │
  │ 4. Adaptive Thresholding: Filters noise, shadows, and creases    │
  │ 5. Bounding Box Extraction: Crops tightly around digit contour   │
  │ 6. Aspect-Ratio Preserving Resize: Fits within 20×20 box         │
  │ 7. Center-of-Mass (Centroid) Alignment: Centers into 28×28 canvas│
  │ 8. Pixel Normalization: Scales [0, 255] -> [0.0, 1.0]            │
  └──────────────────────────────────────────────────────────────────┘
             │
             ▼
      (784-Dimensional Input Vector)
             │
             ▼
  ┌──────────────────────────────────────────────────────────────────┐
  │                 DEEP NEURAL NETWORK ARCHITECTURE                 │
  ├──────────────────────────────────────────────────────────────────┤
  │ Input Layer:    784 Neurons (Flattened 28×28 grayscale pixels)   │
  │ Dense Layer 1:  256 Neurons + He (Kaiming) Normal + ReLU         │
  │ Regularization: Dropout (Rate = 0.2)                             │
  │ Dense Layer 2:  128 Neurons + He (Kaiming) Normal + ReLU         │
  │ Regularization: Dropout (Rate = 0.2)                             │
  │ Output Layer:   10 Neurons (Digits 0 through 9) + Softmax        │
  │ Optimizer:      Adam (Adaptive Moment Estimation, lr = 0.001)    │
  │ Loss Function:  Categorical Cross-Entropy                        │
  └──────────────────────────────────────────────────────────────────┘
             │
             ▼
  ┌──────────────────────────────────────────────────────────────────┐
  │                         OUTPUT DISPLAY                           │
  ├──────────────────────────────────────────────────────────────────┤
  │  • Predicted Digit: (e.g. 7)                                     │
  │  • Prediction Confidence: (e.g. 99.97%)                          │
  │  • Softmax Class Probabilities: Bar chart for digits 0 to 9      │
  │  • Model Accuracy: 97.93% (On 10,000 Unseen MNIST Test Images)   │
  │  • Inference Latency: ~1.2 ms                                    │
  └──────────────────────────────────────────────────────────────────┘
```

---

## 3. Mathematical Formulation

### 3.1 Forward Propagation
For each layer $l$:
$$Z^{[l]} = W^{[l]} A^{[l-1]} + b^{[l]}$$

Where:
- $W^{[l]}$ is the weight matrix initialized using **He (Kaiming) Normal Initialization**:
  $$W \sim \mathcal{N}\left(0, \sqrt{\frac{2}{n_{\text{in}}}}\right)$$
- $b^{[l]}$ is the bias vector initialized to zeros.

### 3.2 Activation Functions
- **Hidden Layers (ReLU):**
  $$A^{[l]} = \text{ReLU}(Z^{[l]}) = \max(0, Z^{[l]})$$
  *Advantage:* Eliminates the vanishing gradient problem, computes rapidly, and induces sparse representations.

- **Output Layer (Softmax):**
  $$P(y = k \mid X) = \frac{e^{Z_k^{[L]} - \max(Z^{[L]})}}{\sum_{j=0}^{9} e^{Z_j^{[L]} - \max(Z^{[L]})}}$$
  *Advantage:* Transforms raw logits into an exact probability distribution where $\sum_{k=0}^9 P_k = 1.0$.

### 3.3 Loss Function (Categorical Cross-Entropy)
$$\mathcal{L} = -\frac{1}{N} \sum_{i=1}^N \sum_{k=0}^9 y_{ik} \log(p_{ik} + \epsilon) + \frac{\lambda}{2N} \sum_l \|W^{[l]}\|_F^2$$
Where $y_{ik} \in \{0, 1\}$ is the one-hot ground-truth label, $p_{ik}$ is the predicted softmax probability, and $\lambda$ is the L2 weight regularization parameter.

### 3.4 Adam Optimizer Updates
Adam calculates exponentially decaying averages of past gradients ($m_t$) and squared gradients ($v_t$):
$$m_t = \beta_1 m_{t-1} + (1 - \beta_1) g_t, \quad v_t = \beta_2 v_{t-1} + (1 - \beta_2) g_t^2$$
$$\hat{m}_t = \frac{m_t}{1 - \beta_1^t}, \quad \hat{v}_t = \frac{v_t}{1 - \beta_2^t}$$
$$\theta_{t+1} = \theta_t - \frac{\alpha}{\sqrt{\hat{v}_t} + \epsilon} \hat{m}_t$$

### 3.5 Center of Mass (Centroid) Image Alignment
To match Yann LeCun's MNIST standard alignment:
$$c_x = \frac{\sum_{x, y} x \cdot I(x, y)}{\sum_{x, y} I(x, y)}, \quad c_y = \frac{\sum_{x, y} y \cdot I(x, y)}{\sum_{x, y} I(x, y)}$$
The digit is shifted such that $(c_x, c_y)$ is positioned at $(13.5, 13.5)$ on the 28×28 grid.

---

## 4. Attached Datasets

All actual dataset files are located in `handwritten_digit_recognition/data/`:
1. `mnist.npz` (11 MB): Complete archive containing 60,000 training images, 60,000 training labels, 10,000 test images, and 10,000 test labels.
2. `mnist_train_5000.csv` (8.7 MB): Tabular CSV dataset containing 5,000 training rows. Columns: `label,pixel0,pixel1,...,pixel783`.
3. `mnist_test_1000.csv` (1.7 MB): Tabular CSV dataset containing 1,000 test rows for laboratory verification.
4. `mnist_sample_1000.csv` (1.7 MB): Direct sample CSV for Excel/Pandas inspection.
5. Official binary files: `train-images-idx3-ubyte.gz`, `train-labels-idx1-ubyte.gz`, `t10k-images-idx3-ubyte.gz`, `t10k-labels-idx1-ubyte.gz`.

---

## 5. Experimental Results & Performance Analysis

### 5.1 Training Summary
- **Dataset:** Official MNIST (60,000 train, 10,000 test)
- **Epochs:** 12 epochs
- **Batch Size:** 128
- **Learning Rate:** 0.001
- **Training Time:** ~7.1 seconds on Apple Silicon
- **Final Validation Accuracy:** 98.46%
- **Final Test Accuracy:** **97.93%** (on 10,000 unseen test samples)

### 5.2 Per-Digit Classification Report
| Digit | Precision (%) | Recall (%) | F1-Score (%) | Test Support |
| :---: | :-----------: | :--------: | :----------: | :----------: |
| **0** | 97.69%        | 99.08%     | 98.38%       | 980          |
| **1** | 98.17%        | 99.30%     | 98.73%       | 1135         |
| **2** | 98.42%        | 96.80%     | 97.61%       | 1032         |
| **3** | 97.36%        | 98.51%     | 97.93%       | 1010         |
| **4** | 97.97%        | 98.47%     | 98.22%       | 982          |
| **5** | 98.31%        | 97.53%     | 97.92%       | 892          |
| **6** | 97.03%        | 98.96%     | 97.98%       | 958          |
| **7** | 97.94%        | 96.89%     | 97.41%       | 1028         |
| **8** | 98.02%        | 96.82%     | 97.42%       | 974          |
| **9** | 98.39%        | 96.83%     | 97.60%       | 1009         |
| **Average** | **97.93%** | **97.92%** | **97.92%** | **10,000** |

---

## 6. How to Run the Project

### Option A: Interactive Web Application (Best for College Presentations)
```bash
python run_digit_app.py
```
Open your browser at `http://localhost:8501`.
- Draw any digit on the interactive canvas.
- Upload any photo taken from a phone or computer.
- Click any quick sample digit (0–9) to test immediately.
- Inspect the live prediction, confidence percentage, probability distribution, and 97.93% model accuracy.

### Option B: Command-Line Single Image Prediction
```bash
python run_predict.py handwritten_digit_recognition/sample_images/photo_paper_digit_7.jpg
```

### Option C: Retrain the Neural Network Model
```bash
cd handwritten_digit_recognition
python train.py
```

### Option D: Run Test Evaluation & Print Full Confusion Matrix
```bash
cd handwritten_digit_recognition
python evaluate.py
```

---

## 7. Viva Voce Questions & Answers (B.Tech CSE External Exam)

#### Q1: What is the MNIST dataset?
**Answer:** MNIST (Modified National Institute of Standards and Technology) is a benchmark dataset of 70,000 28×28 grayscale images of handwritten digits (60,000 training, 10,000 test) created by Yann LeCun, Corinna Cortes, and Christopher Burges in 1998.

#### Q2: Why is the input dimension 784?
**Answer:** Each image in MNIST is 28 pixels wide and 28 pixels high. Flattening the 2D grid into a 1D vector gives $28 \times 28 = 784$ features. Each feature corresponds to the grayscale intensity of a pixel.

#### Q3: Why does a real photograph of a digit need preprocessing before passing to the network?
**Answer:** Real photos have arbitrary dimensions, RGB color channels, light backgrounds with dark ink, uneven shadows, and off-center placement. MNIST expects white digits on a black background centered by center-of-mass in a 20×20 box on a 28×28 canvas with pixel values normalized to $[0.0, 1.0]$. Without preprocessing, the model would fail due to domain shift.

#### Q4: Why is ReLU preferred over Sigmoid in hidden layers?
**Answer:** The derivative of Sigmoid is at most 0.25, which leads to vanishing gradients in deeper networks. ReLU has a constant gradient of 1 for all positive inputs ($f'(z) = 1$ when $z > 0$), allowing gradients to backpropagate without diminishing, and trains much faster.

#### Q5: What is Softmax and why is it used at the final layer?
**Answer:** Softmax normalizes an arbitrary real-valued logit vector into a probability distribution where every value is between 0 and 1, and the sum of all values equals 1.0. This allows us to interpret outputs as class probabilities for digits 0 through 9.

#### Q6: What is Categorical Cross-Entropy Loss?
**Answer:** Cross-entropy measures the divergence between the true distribution (one-hot vector $y$) and predicted probabilities ($\hat{y}$):
$$\mathcal{L} = -\sum_{k=0}^9 y_k \log(\hat{y}_k)$$
When the predicted probability for the correct class is close to 1, the loss approaches 0; if it approaches 0, the loss approaches infinity.

#### Q7: Why use the Adam optimizer instead of vanilla SGD?
**Answer:** Stochastic Gradient Descent (SGD) uses a single learning rate for all parameters. Adam (Adaptive Moment Estimation) maintains individual adaptive learning rates for each weight by calculating first moments (momentum) and second moments (uncentered variance), leading to much faster convergence and stability.

#### Q8: What accuracy was achieved and how was it verified?
**Answer:** Our model achieved **97.93% accuracy** on 10,000 unseen test samples from the official MNIST test partition.

