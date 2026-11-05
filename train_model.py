"""
Email Spam Detection - Model Training Script
Target Audience: B.Tech 2nd Year Mini Project
Algorithm: Multinomial Naive Bayes + TF-IDF Vectorizer
"""

import os
import joblib
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, classification_report

def train_spam_model(data_path="dataset.csv", model_dir="."):
    print("=" * 60)
    print("  EMAIL SPAM CLASSIFIER - TRAINING PIPELINE")
    print("=" * 60)
    
    # 1. Load the dataset
    print(f"\n[1] Loading dataset from: {data_path}")
    df = pd.read_csv(data_path)
    print(f"    Total records: {len(df)}")
    print(f"    Spam count: {len(df[df['Spam/Ham'] == 'Spam'])}")
    print(f"    Ham count : {len(df[df['Spam/Ham'] == 'Ham'])}")
    
    # 2. Preprocess text: Combine Subject and Message
    print("\n[2] Combining 'Subject' and 'Message' into combined text feature...")
    df['Full_Text'] = df['Subject'].astype(str) + " " + df['Message'].astype(str)
    
    X = df['Full_Text']
    y = df['Spam/Ham']
    
    # 3. Text Vectorization using TF-IDF (Term Frequency - Inverse Document Frequency)
    print("\n[3] Converting text to numerical vectors using TfidfVectorizer...")
    # ngram_range=(1, 2) captures single words and 2-word phrases like 'cash reward' or 'meeting scheduled'
    vectorizer = TfidfVectorizer(
        lowercase=True,
        ngram_range=(1, 2),
        stop_words='english'
    )
    X_tfidf = vectorizer.fit_transform(X)
    print(f"    Vocabulary size extracted: {len(vectorizer.get_feature_names_out())} tokens")
    
    # 4. Train Multinomial Naive Bayes Classifier
    print("\n[4] Training Multinomial Naive Bayes model...")
    # alpha=1.0 applies Laplace smoothing to prevent zero-probability problems for unseen words
    model = MultinomialNB(alpha=1.0)
    model.fit(X_tfidf, y)
    
    # 5. Evaluate on training data
    predictions = model.predict(X_tfidf)
    acc = accuracy_score(y, predictions)
    print(f"    Training Accuracy: {acc * 100:.2f}%")
    print("\n    Classification Report:")
    print(classification_report(y, predictions, zero_division=0))
    
    # 6. Save model and vectorizer
    model_file = os.path.join(model_dir, "model.pkl")
    vectorizer_file = os.path.join(model_dir, "vectorizer.pkl")
    
    joblib.dump(model, model_file)
    joblib.dump(vectorizer, vectorizer_file)
    print(f"\n[5] Model artifacts saved successfully:")
    print(f"    - Model     : {model_file}")
    print(f"    - Vectorizer: {vectorizer_file}")
    
    # 7. Quick verification test
    print("\n[6] Running test verification:")
    test_samples = [
        "Win 10000 cash voucher reward today!",
        "Can we reschedule the project viva to Thursday?"
    ]
    for sample in test_samples:
        vec = vectorizer.transform([sample])
        pred = model.predict(vec)[0]
        probs = model.predict_proba(vec)[0]
        prob_dict = dict(zip(model.classes_, probs))
        confidence = prob_dict[pred] * 100
        print(f"    Input: \"{sample}\"")
        print(f"    --> Predicted: [{pred}] (Confidence: {confidence:.2f}%)\n")
        
    print("=" * 60)
    print("Training finished successfully! Ready for web app integration.")
    print("=" * 60)

if __name__ == "__main__":
    train_spam_model()

