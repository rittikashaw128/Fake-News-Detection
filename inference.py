import pickle

# Load model
with open("model.pkl", "rb") as f:
    model = pickle.load(f)

def predict_news(text):
    prediction = model.predict([text])[0]
    prob = model.predict_proba([text]).max()
    return prediction, prob

if __name__ == "__main__":
    sample_text = input("📰 Enter a news article text: ")
    label, confidence = predict_news(sample_text)
    print(f"\n🔎 Prediction: {label} ({confidence:.2f} confidence)")
