from flask import Flask, render_template, request
import pickle
import re
import nltk

from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer

# Flask App
app = Flask(__name__)

# Load Model and Vectorizer
model = pickle.load(open("model/model.pkl", "rb"))

vectorizer = pickle.load(open("model/vectorizer.pkl", "rb"))

# NLP Setup
nltk.download('stopwords')

stemmer = PorterStemmer()

stop_words = set(stopwords.words('english'))

# Text Cleaning Functions
def clean_text(text):

    text = re.sub('[^a-zA-Z]', ' ', text)

    text = text.lower()

    text = text.split()

    text = [
        stemmer.stem(word)
        for word in text
        if word not in stop_words
    ]

    text = " ".join(text)

    return text

# Routes
@app.route("/")
def home():
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():

    news = request.form["news"]

    cleaned_news = clean_text(news)

    vector_input = vectorizer.transform([cleaned_news])

    prediction = model.predict(vector_input)[0]

    print(prediction)

    if int(prediction) == 0:
        result = "Fake News"
    else:
        result = "Real News"

    return render_template(
        "index.html",
        prediction_text=result
    )

# Run App
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)