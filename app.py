from flask import Flask, render_template, request
import pandas as pd
import os
import logging
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)

# Load dataset
df = pd.read_csv("spam.csv", encoding='latin-1')
df = df[['v1', 'v2']]
df.columns = ['label', 'message']
df['label'] = df['label'].map({'ham': 0, 'spam': 1})

# Train model
X_train, X_test, y_train, y_test = train_test_split(
    df['message'], df['label'], test_size=0.2, random_state=42)

vectorizer = TfidfVectorizer()
X_train_tfidf = vectorizer.fit_transform(X_train)

model = MultinomialNB()
model.fit(X_train_tfidf, y_train)
logger.info("Model trained successfully on %d messages", len(df))


@app.route('/')
def home():
    return render_template("index.html", prediction_text=None, original_message="")


@app.route('/predict', methods=['POST'])
def predict():
    try:
        message = request.form.get('message', '').strip()
        logger.info("Received message: %r", message)

        if not message:
            return render_template("index.html", prediction_text="Please enter a message.", original_message="")

        data = vectorizer.transform([message])
        prediction = model.predict(data)[0]
        logger.info("Prediction: %s", prediction)

        result = "Spam Message 🚨" if prediction == 1 else "Not Spam Message ✅"
        return render_template("index.html", prediction_text=result, original_message=message)

    except Exception as e:
        logger.exception("Error during prediction")
        return render_template("index.html", prediction_text=f"Error: {e}", original_message="")


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=True)
