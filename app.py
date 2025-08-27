from flask import Flask, render_template, request
import pickle

# Define app FIRST
app = Flask(__name__)

# Load model and vectorizer
model = pickle.load(open("model.pkl", "rb"))
vectorizer = pickle.load(open("vectorizer.pkl", "rb"))

# Define routes AFTER app is created
@app.route('/', methods=['GET', 'POST'])
def home():
    prediction = None
    warning = None

    if request.method == 'POST':
        text = request.form['news']
        mode = request.form['mode']

        if len(text.strip()) < 5:
            warning = "⚠️ Text is too short to predict accurately."
        else:
            features = vectorizer.transform([text])
            result = model.predict(features)[0]
            prediction = f"This {mode} looks {'✅ REAL' if result == 'REAL' else '❌ FAKE'}."

    return render_template('index.html', prediction=prediction, warning=warning)

# Start the app
if __name__ == '__main__':
    app.run(debug=True)
