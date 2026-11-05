"""
Quick CLI Tester for Email Spam Detection
Zero dependencies - runs using Python standard library!
"""

from standalone_web_app import classifier, RAW_DATA

def main():
    print("==================================================")
    print("   EMAIL SPAM DETECTION - QUICK TERMINAL TEST")
    print("==================================================")
    print(f"Loaded {len(RAW_DATA)} training samples.")
    print("Enter 'exit' or 'quit' at any prompt to stop.\n")

    test_examples = [
        ("Special offer", "Congratulations you won a surprise prize and cash reward today!"),
        ("Lab assignment", "Please review the project submission and meeting time tomorrow."),
        ("Bank alert", "Your account needs attention verify your details now for free reward.")
    ]

    print("--- Running Test Examples ---")
    for subj, msg in test_examples:
        full_text = f"{subj} {msg}"
        pred, probs = classifier.predict(full_text)
        print(f"Subject : {subj}")
        print(f"Message : {msg}")
        print(f"Verdict : [{pred.upper()}] (Spam: {probs['Spam']*100:.1f}%, Ham: {probs['Ham']*100:.1f}%)")
        print("-" * 50)

    print("\n--- Try your own email ---")
    while True:
        try:
            subj = input("\nEnter Subject (or 'exit'): ").strip()
            if subj.lower() in ['exit', 'quit']:
                break
            msg = input("Enter Message: ").strip()
            if msg.lower() in ['exit', 'quit']:
                break
            
            pred, probs = classifier.predict(f"{subj} {msg}")
            print("\n" + "=" * 40)
            if pred == "Spam":
                print(f"🚨 VERDICT: SPAM (Confidence: {probs['Spam']*100:.2f}%)")
            else:
                print(f"🟢 VERDICT: HAM / SAFE (Confidence: {probs['Ham']*100:.2f}%)")
            print("=" * 40)
        except (KeyboardInterrupt, EOFError):
            break

    print("\nThank you for testing the spam classifier!")

if __name__ == "__main__":
    main()

