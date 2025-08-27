import pandas as pd
import pickle
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import PassiveAggressiveClassifier
from sklearn.metrics import accuracy_score, confusion_matrix

def train_model():
   def train_model():
    # Load static datasets
    true_df = pd.read_csv("True.csv")
    fake_df = pd.read_csv("Fake.csv")

    # Add labels
    true_df["label"] = "REAL"
    fake_df["label"] = "FAKE"

    # ✅ Fetch fresh live news
    live_news = fetch_live_news("your_newsapi_key_here", count=100)
    live_df = pd.DataFrame(live_news, columns=["text"])
    live_df["label"] = "REAL"   # because API gives real headlines

    # ✅ Merge datasets (old + live)
    df = pd.concat([true_df, live_df, fake_df]).sample(frac=1, random_state=42).reset_index(drop=True)

    # Continue same pipeline...
    X_train, X_test, y_train, y_test = train_test_split(df["text"], df["label"], test_size=0.2, random_state=42)
    vectorizer = TfidfVectorizer(stop_words="english", max_df=0.7)
    X_train_vec = vectorizer.fit_transform(X_train)
    X_test_vec = vectorizer.transform(X_test)

    pac = PassiveAggressiveClassifier(max_iter=50)
    pac.fit(X_train_vec, y_train)

    y_pred = pac.predict(X_test_vec)
    print(f"✅ Accuracy: {accuracy_score(y_test, y_pred)*100:.2f}%")
    print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))

    # Save model + vectorizer
    pickle.dump(pac, open("model.pkl", "wb"))
    pickle.dump(vectorizer, open("vectorizer.pkl", "wb"))
    print("📁 Model retrained with live data and saved!")


def run_inference():
    # Load model & vectorizer
    with open("model.pkl", "rb") as f:
        model = pickle.load(f)
    with open("vectorizer.pkl", "rb") as f:
        vectorizer = pickle.load(f)

    print("\n📰 Fake News Detector (type 'exit' to quit)\n")
    while True:
        user_text = input("Enter a news article to check if it's REAL or FAKE: ")
        if user_text.lower().strip() == "exit":
            print("👋 Exiting...")
            break

        # ✅ Allow short inputs
        user_features = vectorizer.transform([user_text])
        prediction = model.predict(user_features)[0]

        if len(user_text.split()) < 5:
            print("⚠ Short text, prediction may be less accurate.")
        
        print("✅ This news looks REAL.\n" if prediction == "REAL" else "❌ This news looks FAKE.\n")

if __name__ == "__main__":
    # First run training once (comment it after training to save time)
    train_model()
    run_inference()