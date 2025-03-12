from flask import Flask, render_template, request, jsonify
import joblib

app = Flask(__name__)

# Load model and vectorizer
model = joblib.load("sentiment_model.pkl")
vectorizer = joblib.load("vectorizer.pkl")

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    text = request.form['text']
    text_vectorized = vectorizer.transform([text])
    prediction = model.predict(text_vectorized)[0]

    sentiment_map = {0: "Negative", 1: "Positive", 2: "Neutral"}
    return jsonify({'sentiment': sentiment_map.get(prediction, "Unknown")})

if __name__ == '__main__':
    app.run(debug=True)
