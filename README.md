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

## 🎓 B.Tech Viva Voce Preparation Guide (Top 10 Q&A)

### Q1: What is the difference between "Spam" and "Ham"?
**Answer:**  
- **Spam:** Unsolicited, commercial, scam, phishing, or unwanted bulk emails.
- **Ham:** Genuine, authentic, desired emails from legitimate senders (e.g., college notices, bills, personal chats).

### Q2: Why is the algorithm called "Naive"?
**Answer:**  
It is called "naive" because it makes the strong assumption of **conditional independence** between every pair of features (words) given the class label. In reality, words like *"cash"* and *"reward"* frequently appear together, but the algorithm treats them as completely independent. Despite this assumption, it performs remarkably well in practice.

### Q3: Why is Multinomial Naive Bayes preferred over Gaussian Naive Bayes for text classification?
**Answer:**  
- **Gaussian Naive Bayes** assumes features follow a continuous normal (bell-curve) distribution (e.g., height, temperature).
- **Multinomial Naive Bayes** models discrete counts (word occurrences or TF-IDF values in text documents), which directly matches natural language data.

### Q4: What is TF-IDF and why use it?
**Answer:**  
TF-IDF stands for **Term Frequency - Inverse Document Frequency**:
$$\text{TF-IDF}(w, d, D) = \text{TF}(w, d) \times \log\left(\frac{N}{\text{DF}(w)}\right)$$
It ensures that common words appearing in every document (e.g., "the", "is") receive a low weight, while rare and informative keywords (e.g., "prize", "voucher", "assignment") receive high weights.

### Q5: What is the "Zero Frequency Problem" and how is it resolved?
**Answer:**  
If a testing email contains a word not present in the training set for a particular class, the estimated probability $P(w \mid C)$ is $0$. Because probabilities are multiplied together, a single zero wipes out the entire score. We solve this using **Laplace Smoothing** by adding $1$ to the numerator and $|V|$ to the denominator.

### Q6: What are the evaluation metrics for spam detection?
**Answer:**  
1. **Accuracy:** Overall correctness: $\frac{TP + TN}{TP + TN + FP + FN}$
2. **Precision:** Out of all emails marked as spam, how many were actually spam? (Crucial: high precision prevents important emails from going to spam!).
3. **Recall:** Out of all actual spam emails, how many did the system catch?
4. **F1-Score:** Harmonic mean of precision and recall.

### Q7: In email filtering, is False Positive worse or False Negative?
**Answer:**  
**False Positive is far worse!**  
- **False Negative:** A spam email reaches the inbox (annoying, but user can delete it).
- **False Positive:** A critical email (e.g., college admission offer, exam schedule, job interview) is misclassified as spam and missed by the user.

### Q8: What are N-grams?
**Answer:**  
An N-gram is a continuous sequence of $N$ items from a text.
- Unigram ($N=1$): "free", "gift"
- Bigram ($N=2$): "free gift", "cash reward", "project meeting"  
Using bigrams helps the model preserve context that single words miss.

### Q9: Can an attacker bypass this spam filter?
**Answer:**  
Yes, spammers use evasion techniques such as:
- Misspellings ("W!N", "fr33", "pr1ze")
- Inserting random benign words ("good word attack")
- Embedding text inside images instead of plaintext  
Modern filters combine Naive Bayes with DKIM/SPF domain verification, sender reputation scores, and deep learning transformers (BERT).

### Q10: What are the future enhancements for this project?
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

