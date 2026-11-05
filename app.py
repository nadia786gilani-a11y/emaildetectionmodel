"""
Email Spam Detection Web App
Designed for B.Tech Mini Project / Demo
Framework: Streamlit + Scikit-Learn (Multinomial Naive Bayes & TF-IDF)
"""

import os
import streamlit as st
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB

# ---------------------------------------------------------
# Page Configuration
# ---------------------------------------------------------
st.set_page_config(
    page_title="Email Spam Classifier | B.Tech Project",
    page_icon="📧",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ---------------------------------------------------------
# Custom Styling (CSS)
# ---------------------------------------------------------
st.markdown("""
<style>
    .main-header {
        font-size: 2.2rem;
        font-weight: 700;
        color: #1E3A8A;
        margin-bottom: 0.2rem;
    }
    .sub-header {
        font-size: 1.05rem;
        color: #64748B;
        margin-bottom: 1.5rem;
    }
    .spam-box {
        background-color: #FEF2F2;
        border: 2px solid #EF4444;
        border-radius: 10px;
        padding: 20px;
        color: #991B1B;
        text-align: center;
        font-size: 1.3rem;
        font-weight: 700;
    }
    .ham-box {
        background-color: #F0FDF4;
        border: 2px solid #22C55E;
        border-radius: 10px;
        padding: 20px;
        color: #166534;
        text-align: center;
        font-size: 1.3rem;
        font-weight: 700;
    }
    .metric-card {
        background: #F8FAFC;
        border: 1px solid #E2E8F0;
        border-radius: 8px;
        padding: 12px;
        text-align: center;
    }
    .keyword-pill {
        display: inline-block;
        background-color: #E2E8F0;
        color: #334155;
        padding: 4px 10px;
        border-radius: 12px;
        margin: 3px;
        font-size: 0.85rem;
    }
</style>
""", unsafe_allow_html=True)

# ---------------------------------------------------------
# Load Dataset & Train/Cache Model
# ---------------------------------------------------------
DATASET_PATH = "dataset.csv"

@st.cache_resource
def load_data_and_train():
    if not os.path.exists(DATASET_PATH):
        # Fallback dataset if file is missing
        data = {
            "Subject": [
                "Congratulations! You won a prize",
                "Urgent account notification",
                "Free shopping voucher",
                "You are a lucky winner",
                "Exclusive cash reward",
                "Meeting scheduled",
                "College assignment reminder",
                "Monthly electricity bill",
                "Project discussion",
                "Birthday invitation"
            ],
            "Message": [
                "You have won a free gift. Claim your prize today!",
                "Your account needs attention. Verify your details.",
                "Get a free shopping voucher worth 5000 rupees. Limited offer!",
                "You have been selected for a special reward.",
                "Claim your surprise cash reward now.",
                "Our project meeting is scheduled for Monday at 10 AM.",
                "Please submit your machine learning assignment by Friday.",
                "Your electricity bill is available. Please check your account.",
                "Can we discuss the project progress tomorrow?",
                "You are invited to my birthday party this Saturday."
            ],
            "Spam/Ham": [
                "Spam", "Spam", "Spam", "Spam", "Spam",
                "Ham", "Ham", "Ham", "Ham", "Ham"
            ]
        }
        df = pd.DataFrame(data)
        df.to_csv(DATASET_PATH, index=False)
    else:
        df = pd.read_csv(DATASET_PATH)

    # Combine Subject + Message
    df["Full_Text"] = df["Subject"].astype(str) + " " + df["Message"].astype(str)
    
    # TF-IDF Vectorizer
    vectorizer = TfidfVectorizer(lowercase=True, ngram_range=(1, 2), stop_words="english")
    X = vectorizer.fit_transform(df["Full_Text"])
    y = df["Spam/Ham"]
    
    # Multinomial Naive Bayes model
    model = MultinomialNB(alpha=1.0)
    model.fit(X, y)
    
    return df, vectorizer, model

df, vectorizer, model = load_data_and_train()

# ---------------------------------------------------------
# Sidebar Controls
# ---------------------------------------------------------
with st.sidebar:
    st.image("https://img.icons8.com/color/96/shield.png", width=70)
    st.title("Project Details")
    st.markdown("""
    **Course:** B.Tech Computer Science / AI-ML  
    **Year:** 2nd Year Mini Project  
    **Topic:** Natural Language Processing (NLP)  
    **Model:** Multinomial Naive Bayes  
    **Feature Extractor:** TF-IDF Vectorizer  
    """)
    st.divider()

    st.subheader("💡 Quick Test Presets")
    st.write("Click any preset to auto-fill the input fields:")
    
    if st.button("🎁 Load Spam: Prize Winner"):
        st.session_state["input_subject"] = "Congratulations! You won a prize"
        st.session_state["input_message"] = "You have won a free gift. Claim your prize today!"

    if st.button("💳 Load Spam: Free Voucher"):
        st.session_state["input_subject"] = "Free shopping voucher"
        st.session_state["input_message"] = "Get a free shopping voucher worth 5000 rupees. Limited offer!"

    if st.button("📚 Load Ham: Assignment Reminder"):
        st.session_state["input_subject"] = "College assignment reminder"
        st.session_state["input_message"] = "Please submit your machine learning assignment by Friday."

    if st.button("👥 Load Ham: Project Meeting"):
        st.session_state["input_subject"] = "Meeting scheduled"
        st.session_state["input_message"] = "Our project meeting is scheduled for Monday at 10 AM."

    st.divider()
    st.caption("Developed for Academic Mini-Project Submission")

# ---------------------------------------------------------
# Main App Header
# ---------------------------------------------------------
st.markdown('<div class="main-header">📧 Email Spam Detection System</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-header">Classify emails in real-time as Spam or Ham (Legitimate) using Natural Language Processing & Machine Learning</div>', unsafe_allow_html=True)

# ---------------------------------------------------------
# Navigation Tabs
# ---------------------------------------------------------
tab1, tab2, tab3 = st.tabs(["🔍 Live Detector", "📊 Dataset & Analysis", "🎓 Viva & Theory Guide"])

# =========================================================
# TAB 1: LIVE DETECTOR
# =========================================================
with tab1:
    col1, col2 = st.columns([1.1, 0.9], gap="large")

    with col1:
        st.subheader("📝 Input Email Content")
        
        default_subject = st.session_state.get("input_subject", "")
        default_message = st.session_state.get("input_message", "")

        subject_input = st.text_input(
            "Email Subject:",
            value=default_subject,
            placeholder="e.g. You have won a lottery ticket!"
        )

        message_input = st.text_area(
            "Email Message Body:",
            value=default_message,
            height=140,
            placeholder="e.g. Dear customer, claim your free reward cash immediately by verifying your details."
        )

        classify_btn = st.button("🚀 Analyze Email", type="primary", use_container_width=True)

    with col2:
        st.subheader("🎯 Classification Results")

        if classify_btn or (subject_input and message_input):
            full_text = (subject_input + " " + message_input).strip()
            
            if not full_text:
                st.warning("⚠️ Please enter a Subject or Message to classify.")
            else:
                # Transform text
                vec_text = vectorizer.transform([full_text])
                prediction = model.predict(vec_text)[0]
                probabilities = model.predict_proba(vec_text)[0]
                classes = list(model.classes_)
                
                spam_idx = classes.index("Spam") if "Spam" in classes else 1
                ham_idx = classes.index("Ham") if "Ham" in classes else 0

                spam_prob = probabilities[spam_idx] * 100
                ham_prob = probabilities[ham_idx] * 100

                # Display Verdict
                if prediction == "Spam":
                    st.markdown("""
                    <div class="spam-box">
                        🚨 SPAM DETECTED<br>
                        <span style="font-size: 0.95rem; font-weight: normal; color: #7F1D1D;">
                            This email appears unsolicited or promotional.
                        </span>
                    </div>
                    """, unsafe_allow_html=True)
                else:
                    st.markdown("""
                    <div class="ham-box">
                        🟢 HAM (LEGITIMATE EMAIL)<br>
                        <span style="font-size: 0.95rem; font-weight: normal; color: #14532D;">
                            This email appears genuine and safe.
                        </span>
                    </div>
                    """, unsafe_allow_html=True)

                st.write("")
                # Confidence Progress Bars
                st.markdown("**Spam Confidence:**")
                st.progress(float(spam_prob / 100))
                st.write(f"**{spam_prob:.2f}%**")

                st.markdown("**Ham Confidence:**")
                st.progress(float(ham_prob / 100))
                st.write(f"**{ham_prob:.2f}%**")

                # Keyword Highlighting
                st.write("---")
                st.markdown("**Detected Tokens Matching Vocabulary:**")
                words = full_text.lower().split()
                matched_tokens = [w for w in set(words) if w in vectorizer.vocabulary_]
                if matched_tokens:
                    pills_html = "".join([f'<span class="keyword-pill">{token}</span>' for token in matched_tokens])
                    st.markdown(pills_html, unsafe_allow_html=True)
                else:
                    st.caption("No exact keyword matches in the current training vocabulary.")
        else:
            st.info("Enter an email subject and body on the left and click **'Analyze Email'**, or select a preset from the sidebar.")

# =========================================================
# TAB 2: DATASET & ANALYSIS
# =========================================================
with tab2:
    st.subheader("📋 Training Database")
    st.write("This application is trained on the following curated dataset:")

    col_a, col_b, col_c = st.columns(3)
    col_a.metric("Total Emails", len(df))
    col_b.metric("Spam Count", len(df[df["Spam/Ham"] == "Spam"]))
    col_c.metric("Ham Count", len(df[df["Spam/Ham"] == "Ham"]))

    st.dataframe(
        df[["Subject", "Message", "Spam/Ham"]],
        use_container_width=True,
        hide_index=True
    )

    st.write("---")
    st.subheader("🔍 Class Breakdown")
    chart_data = df["Spam/Ham"].value_counts().reset_index()
    chart_data.columns = ["Category", "Count"]
    st.bar_chart(chart_data, x="Category", y="Count", color="Category")

# =========================================================
# TAB 3: B.TECH VIVA & THEORY GUIDE
# =========================================================
with tab3:
    st.subheader("🎓 B.Tech Mini Project Viva Guide")
    st.write("Common viva questions and technical explanations to prepare for academic presentation:")

    with st.expander("1. What algorithm is used and why?", expanded=True):
        st.markdown("""
        - **Algorithm:** Multinomial Naive Bayes (MNB) classifier with TF-IDF vectorization.
        - **Why Naive Bayes for text?**
          1. It performs exceptionally well on text classification and high-dimensional sparse data.
          2. It requires minimal computational power and trains extremely fast.
          3. It works reliably even with small training datasets.
        """)

    with st.expander("2. What is Bayes' Theorem formula?", expanded=False):
        st.latex(r"P(C \mid D) = \frac{P(D \mid C) \cdot P(C)}{P(D)}")
        st.markdown("""
        Where:
        - $P(C \mid D)$ = Posterior Probability: Probability of class $C$ (Spam or Ham) given document $D$.
        - $P(D \mid C)$ = Likelihood: Probability that document $D$ is generated given class $C$.
        - $P(C)$ = Prior Probability: Overall probability of class $C$ in the training data.
        - $P(D)$ = Marginal Probability: Evidence/normalizing constant.
        """)

    with st.expander("3. Why is it called 'Naive'?", expanded=False):
        st.markdown("""
        It is called **'Naive'** because it assumes that every word in the document is **conditionally independent** of every other word given the class label. 
        In real human language, words often depend on context (e.g. 'free' followed by 'gift'), but despite this naive simplification, Naive Bayes works remarkably well in practice.
        """)

    with st.expander("4. What is the difference between CountVectorizer and TF-IDF?", expanded=False):
        st.markdown("""
        - **CountVectorizer (Bag of Words):** Simply counts how many times each word occurs in the text.
        - **TF-IDF (Term Frequency - Inverse Document Frequency):** Penalizes words that appear everywhere across all documents (like 'the', 'is', 'a') and awards higher weights to rare, discriminative words (like 'voucher', 'reward', 'assignment').
        """)

    with st.expander("5. What is Laplace Smoothing (alpha = 1.0)?", expanded=False):
        st.markdown("""
        If a new email contains a word that was never seen during training, its probability $P(w \mid C)$ would become $0$.
        Since Naive Bayes multiplies word probabilities together, a single $0$ would make the whole probability $0$.
        **Laplace Smoothing** adds $+1$ to numerator and $+|V|$ (vocabulary size) to denominator to ensure no probability ever drops to zero.
        """)

    with st.expander("6. How to scale this project for final year submission?", expanded=False):
        st.markdown("""
        - Use larger public datasets like the **Enron Email Dataset** or **SMS Spam Collection** (5,574 messages).
        - Add Lemmatization (using NLTK or spaCy).
        - Compare Naive Bayes with Support Vector Machines (SVM), Logistic Regression, and BERT / Transformers.
        - Add a database backend (SQLite / PostgreSQL) to save real-time spam logs.
        """)

