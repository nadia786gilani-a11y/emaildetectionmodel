# 📧 Email Spam Detection System
**B.Tech 2nd Year Mini Project — Natural Language Processing & Machine Learning**

---

## 📌 Project Overview
The **Email Spam Detection System** is an automated text classification application designed to categorize incoming emails as either **Spam** (unsolicited, fraudulent, or promotional messages) or **Ham** (legitimate, personal, or official emails).

This project demonstrates the core concepts taught in undergraduate computer science and data science curricula:
- **Text Preprocessing** (Lowercasing, Tokenization, Stopword Filtering)
- **Feature Extraction** (Bag-of-Words & TF-IDF Vectorization)
- **Probabilistic Modeling** (Multinomial Naive Bayes Classifier)
- **Laplace Smoothing** to handle unseen vocabulary
- **Interactive UI Deployment** (Streamlit & Zero-dependency Pure Python Web Server)

---

## 📊 Dataset Description
The project is built and evaluated on the following labeled dataset:

| # | Subject | Message | Label |
|---|---|---|---|
| 1 | Congratulations! You won a prize | You have won a free gift. Claim your prize today! | **Spam** |
| 2 | Urgent account notification | Your account needs attention. Verify your details. | **Spam** |
| 3 | Free shopping voucher | Get a free shopping voucher worth 5000 rupees. Limited offer! | **Spam** |
| 4 | You are a lucky winner | You have been selected for a special reward. | **Spam** |
| 5 | Exclusive cash reward | Claim your surprise cash reward now. | **Spam** |
| 6 | Meeting scheduled | Our project meeting is scheduled for Monday at 10 AM. | **Ham** |
| 7 | College assignment reminder | Please submit your machine learning assignment by Friday. | **Ham** |
| 8 | Monthly electricity bill | Your electricity bill is available. Please check your account. | **Ham** |
| 9 | Project discussion | Can we discuss the project progress tomorrow? | **Ham** |
| 10| Birthday invitation | You are invited to my birthday party this Saturday. | **Ham** |

---

## 🧠 System Architecture & Mathematical Foundation

### 1. Bayes' Theorem
The classifier calculates the posterior probability of a class $C \in \{\text{Spam}, \text{Ham}\}$ given an email text $D$:

$$P(C \mid D) = \frac{P(D \mid C) \cdot P(C)}{P(D)}$$

Since $P(D)$ is constant for both classes, we maximize:

$$P(C \mid D) \propto P(C) \prod_{i=1}^{n} P(w_i \mid C)$$

### 2. Laplace Smoothing ($\alpha = 1$)
If an email contains an unfamiliar word, $P(w \mid C)$ would normally equal $0$, causing the entire product to collapse to zero. Laplace smoothing prevents this:

$$P(w \mid C) = \frac{\text{Count}(w, C) + \alpha}{\sum_{w' \in V} \text{Count}(w', C) + \alpha \cdot |V|}$$

Where $|V|$ is the total vocabulary size and $\alpha = 1.0$.

---

## 🚀 How to Run the Applications

You have **three ways** to run and demonstrate this project:

### 🌟 Option 1: Standalone Web App (ZERO Dependencies!)
> **Best for college lab computers with restricted internet or permissions!**  
> Uses only Python's built-in standard library (`http.server`, `json`, `math`). No `pip install` required!

```bash
# In your terminal:
python3 standalone_web_app.py
```
Then open your browser and navigate to:
```
http://localhost:8080
```

---

### 🎨 Option 2: Modern Streamlit Web Dashboard
> **Best for project submissions, slides, and laptop presentations!**  
> Features live interactive sliders, gauge bars, tabs, and dataset visualizations.

1. Create a virtual environment and install dependencies:
```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

2. Train the model (optional, app auto-trains if missing):
```bash
python3 train_model.py
```

3. Launch the Streamlit application:
```bash
streamlit run app.py
```
The app will automatically open in your default browser at `http://localhost:8501`.

---

### 💻 Option 3: Terminal / CLI Quick Tester
> Run instant test cases directly inside your terminal:
```bash
python3 cli_test.py
```

---


##: What are the future enhancements for this project?
**Answer:**  
- Train on larger benchmarks (e.g., Enron 30,000+ emails or SMS Spam collection).
- Add support for email attachment scanning and sender reputation checking.
- Deploy as a Chrome extension or Gmail API sidecar filter.

---

## 📁 Project Directory Structure
```
email_spam_detection/
├── dataset.csv             # The 10 provided email records
├── standalone_web_app.py   # Pure Python web server (zero dependencies)
├── app.py                  # Streamlit web app interface
├── train_model.py          # Scikit-learn training & model serialization
├── cli_test.py             # Interactive terminal tester
├── requirements.txt        # Python package dependencies
└── README.md               # Complete documentation and viva guide
```

