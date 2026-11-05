# Handwritten Digit Recognition Using Deep Learning (MNIST)

**Student:** Nadia (2nd Year B.Tech Computer Science & Engineering)  
**Dataset:** Official MNIST (60,000 Training, 10,000 Testing Samples)  
**Model Performance:** **97.93% Test Accuracy** on 10,000 Unseen Test Digits  

---

## Quick Start (1-Click Run)

### 1. Launch the Interactive Web Application
```bash
python run_digit_app.py
```
Open your browser at **`http://localhost:8501`** to:
- Draw digits on the interactive touch/mouse canvas.
- Take or upload photos of handwritten digits from your phone or PC.
- Test quick-click sample pictures of digits 0 through 9.
- See the predicted digit, confidence score, full probability distribution, and 97.93% model accuracy!

---

### 2. Predict Any Picture from Terminal (CLI)
```bash
python run_predict.py handwritten_digit_recognition/sample_images/photo_paper_digit_7.jpg
```
Output:
```
=================================================================
  ANALYZING HANDWRITTEN DIGIT PICTURE: photo_paper_digit_7.jpg
=================================================================
           █████ 
              █  
             █   
            █    
            █    

   >>> PREDICTED DIGIT:  7
   >>> CONFIDENCE:       99.97%
   >>> MODEL ACCURACY:   97.93% (Evaluated on 10,000 MNIST Test Images)
-----------------------------------------------------------------
Digit Probability Distribution:
   Digit 7: [███████████████████████████████████████░]  99.97% <== PREDICTED
```

---

### 3. Evaluate the Entire 10,000 Test Set
```bash
cd handwritten_digit_recognition
python evaluate.py
```

---

### 4. Retrain the Neural Network Model
```bash
cd handwritten_digit_recognition
python train.py
```

---

## Project Structure & Attached Dataset

```
handwritten_digit_recognition/
├── data/                                 <-- ATTACHED ACTUAL DATASETS
│   ├── mnist.npz                         <-- Full 60,000 train + 10,000 test set (11 MB)
│   ├── mnist_train_5000.csv              <-- 5,000 sample CSV for Excel / College Labs
│   ├── mnist_test_1000.csv               <-- 1,000 sample CSV for testing
│   ├── mnist_sample_1000.csv             <-- Sample CSV with 784 pixel columns
│   ├── train-images-idx3-ubyte.gz        <-- Official raw binary train images
│   ├── train-labels-idx1-ubyte.gz        <-- Official raw binary train labels
│   ├── t10k-images-idx3-ubyte.gz         <-- Official raw binary test images
│   └── t10k-labels-idx1-ubyte.gz         <-- Official raw binary test labels
├── sample_images/                        <-- REAL PICTURES FOR TESTING
│   ├── photo_paper_digit_0.jpg           <-- Realistic smartphone photos on paper
│   ├── photo_paper_digit_2.jpg
│   ├── photo_paper_digit_3.jpg
│   ├── photo_paper_digit_5.jpg
│   ├── photo_paper_digit_7.jpg
│   ├── photo_paper_digit_8.jpg
│   ├── mnist_digit_0.png to 9.png        <-- Standard MNIST digit images
├── models/
│   ├── digit_model_weights.npz           <-- Trained Deep Neural Network weights
│   └── model_metrics.json                <-- Accuracy & training curves
├── neural_network.py                     <-- Deep MLP Architecture (784 -> 256 -> 128 -> 10)
├── image_preprocessor.py                 <-- Smart Photo Preprocessor (Invert, Center, Crop)
├── train.py                              <-- Training script with Adam optimizer
├── evaluate.py                           <-- Computes test accuracy & confusion matrix
├── predict.py                            <-- CLI Prediction engine
├── app.py                                <-- Interactive Web Application
├── train_pytorch.py                      <-- Alternative PyTorch CNN implementation
├── export_csv_dataset.py                 <-- Exporter from binary to CSV format
├── PROJECT_REPORT.md                     <-- Complete B.Tech CSE College Submission Report
└── README.md                             <-- Quick reference guide
```

---

## Deep Neural Network Architecture
- **Input Layer:** 784 Neurons (28×28 pixels normalized to [0.0, 1.0])
- **Hidden Layer 1:** 256 Neurons + He Normal Initialization + ReLU + Dropout (0.2)
- **Hidden Layer 2:** 128 Neurons + He Normal Initialization + ReLU + Dropout (0.2)
- **Output Layer:** 10 Neurons (Digits 0–9) + Softmax Activation
- **Optimization:** Adam Optimizer ($\alpha=0.001, \beta_1=0.9, \beta_2=0.999$)
- **Loss Function:** Categorical Cross-Entropy Loss
- **Test Accuracy:** **97.93%** (on 10,000 unseen test samples)

